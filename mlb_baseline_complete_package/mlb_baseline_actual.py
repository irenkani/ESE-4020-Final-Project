"""Reproducible MLB attendance baseline; run from any working directory.

Uses only local attendance_mlbdata.csv. 2025 is deliberately never loaded into
model input or evaluated: it remains the final untouched test year.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
SEED = 42
rng = np.random.default_rng(SEED)
N_BOOT = 2000
CATEGORICAL = ['home_team', 'away_team', 'month', 'day_of_week']
ALPHAS = np.logspace(-3, 3, 13)
FOLD_YEARS = [2015, 2016, 2017, 2018, 2019, 2022, 2023]

raw = pd.read_csv(ROOT/'attendance_mlbdata.csv', low_memory=False)
raw['game_date'] = pd.to_datetime(raw['game_date'], errors='coerce')
raw['attendance'] = pd.to_numeric(raw['attendance'], errors='coerce')
counts = {'raw_rows': int(len(raw))}
# Exclude spring training, postseason, postponed/cancelled games, 2020/2021 pandemic seasons.
x = raw.loc[(raw['game_type'] == 'R') & raw['status'].isin(['Final', 'Completed Early'])].copy()
counts['regular_completed_rows_before_dedup'] = int(len(x))
counts['duplicate_game_id_rows_removed'] = int(x['game_id'].duplicated().sum())
# Suspended games can have two date records with identical game IDs/attendance.
# Use the later completion-date record for a single game observation.
x = x.sort_values(['game_id', 'game_date'], kind='mergesort').drop_duplicates('game_id', keep='last')
counts['completed_unique_regular_games'] = int(len(x))
counts['missing_or_nonpositive_attendance_removed'] = int((x['attendance'].isna() | (x['attendance'] <= 0)).sum())
x = x.loc[x['attendance'].notna() & (x['attendance'] > 0)].copy()
counts['pandemic_2020_2021_rows_removed'] = int(x['year'].isin([2020, 2021]).sum())
x = x.loc[~x['year'].isin([2020,2021])].copy()
# Limit to eligible historical seasons, with 2025 segregated and not evaluated.
counts['post_clean_all_years_2008_2025'] = int(len(x))
counts['2025_held_out_rows'] = int((x['year'] == 2025).sum())
x = x.loc[(x['year'] >= 2008) & (x['year'] <= 2024)].copy()
counts['train_through_2023'] = int((x['year'] <= 2023).sum())
counts['2024_validation'] = int((x['year'] == 2024).sum())
counts['training_2024_total'] = int(len(x))
counts['2024_unique_dates'] = int(x.loc[x['year']==2024,'game_date'].nunique())
for col in CATEGORICAL:
    x[col] = x[col].fillna('Unknown').astype(str)
counts['raw_features'] = len(CATEGORICAL)

train = x.loc[x.year <= 2023].copy()
val = x.loc[x.year == 2024].copy()
assert train.game_date.max() < val.game_date.min()
assert not x.game_id.duplicated().any()

# Forward-chaining: train on seasons earlier than fold year; evaluate on that year.
cv_rows=[]
for a in ALPHAS:
    fold_scores=[]
    for yr in FOLD_YEARS:
        tr = train[train.year < yr]
        te = train[train.year == yr]
        if len(te)==0 or len(tr)==0: continue
        prep=ColumnTransformer([('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)])
        model=Pipeline([('preprocess',prep),('ridge',Ridge(alpha=float(a), solver='lsqr', tol=1e-7))])
        model.fit(tr[CATEGORICAL],tr.attendance)
        pred=model.predict(te[CATEGORICAL])
        fold_scores.append(float(mean_absolute_error(te.attendance,pred)))
    cv_rows.append({'alpha':float(a),'mean_cv_mae':float(np.mean(fold_scores)),
                    'fold_mae':dict(zip(map(str,FOLD_YEARS),fold_scores))})
chosen=min(cv_rows,key=lambda v:v['mean_cv_mae'])
alpha=chosen['alpha']
model=Pipeline([('preprocess',ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])),
                ('ridge',Ridge(alpha=alpha, solver='lsqr', tol=1e-7))])
model.fit(train[CATEGORICAL],train.attendance)
pred_ridge=model.predict(val[CATEGORICAL])
counts['encoded_features'] = int(model.named_steps['preprocess'].transform(val[CATEGORICAL]).shape[1])
# Historical benchmark: ONLY 2023 home-team attendance means used on 2024.
prior = train.loc[train.year==2023].groupby('home_team',observed=True)['attendance'].mean()
missing_teams=sorted(set(val.home_team.unique()).difference(prior.index))
if missing_teams:
    raise ValueError('Missing 2023 averages for 2024 home teams: '+repr(missing_teams))
pred_hist=val.home_team.map(prior).to_numpy(dtype=float)
y = val.attendance.to_numpy(dtype=float)

def metrics(p):
    return {'mae':float(mean_absolute_error(y,p)),
            'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p))}

# Paired date-block bootstrap. Each sampled date contributes all games on that date.
date_codes, dates = pd.factorize(val.game_date)
date_indexes=[np.flatnonzero(date_codes==i) for i in range(len(dates))]
day_sizes=np.array([len(ix) for ix in date_indexes],dtype=int)
day_ridge_abs=np.array([np.abs(y[ix]-pred_ridge[ix]).sum() for ix in date_indexes])
day_hist_abs=np.array([np.abs(y[ix]-pred_hist[ix]).sum() for ix in date_indexes])
boots = rng.integers(0,len(dates),size=(N_BOOT,len(dates)))
n = day_sizes[boots].sum(axis=1)
boots_ridge=day_ridge_abs[boots].sum(axis=1)/n
boots_hist=day_hist_abs[boots].sum(axis=1)/n
boots_improvement=boots_hist-boots_ridge
ci=lambda v: [float(z) for z in np.percentile(v,[2.5,97.5])]
result={
 'dataset':counts,
 'split':{'training':'2008–2023 excluding 2020 and 2021','validation':'2024',
          'final_untouched':'2025','cv_folds':FOLD_YEARS,'seed':SEED},
 'ridge':{'chosen_alpha':alpha,'validation':metrics(pred_ridge), 'mae_95ci':ci(boots_ridge)},
 'historical':{'predictor':'2023 home-team mean attendance','validation':metrics(pred_hist),
               'mae_95ci':ci(boots_hist)},
 'comparison':{'mae_improvement_historical_minus_ridge':float(mean_absolute_error(y,pred_hist)-mean_absolute_error(y,pred_ridge)),
               'difference_95ci':ci(boots_improvement),
               'meaningful_improvement':bool(ci(boots_improvement)[0]>0),
               'draws':N_BOOT, 'bootstrap_unit':'game date'},
 'cv_grid':cv_rows,
 'source_file':'attendance_mlbdata.csv'
}
(ROOT/'mlb_baseline_results.json').write_text(json.dumps(result,indent=2)+'\n')
# Produces human-auditable 2024 predictions, no test-year data.
val_output=val[['game_id','game_date','home_team','away_team','attendance']].copy()
val_output['historical_prediction']=pred_hist
val_output['ridge_prediction']=pred_ridge
val_output.to_csv(ROOT/'mlb_2024_validation_predictions.csv',index=False)
print(json.dumps({k:v for k,v in result.items() if k!='cv_grid'},indent=2))
print('Top grid entries:')
for v in sorted(cv_rows,key=lambda z:z['mean_cv_mae'])[:5]: print(v['alpha'],v['mean_cv_mae'])
