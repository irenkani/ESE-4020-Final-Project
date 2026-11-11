"""Reproducible Retrosheet MLB attendance baseline; does not touch 2025 outcomes.
Usage: python run_baseline.py --gameinfo data/gameinfo.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ALPHAS = np.logspace(-3, 3, 13)
SEED = 42
CATEGORICAL = ['hometeam', 'visteam', 'month', 'weekday']
# Exclude 2020/2021 from primary analysis; 2025 is NEVER loaded into model validation.
EXCLUDED_YEARS = {2020, 2021}


def make_model(alpha):
    return Pipeline([
        ('preprocess', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ], remainder='drop')),
        ('ridge', Ridge(alpha=float(alpha)))
    ])


def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    required = ['gid', 'date', 'hometeam', 'visteam', 'attendance', 'gametype']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns {missing}; found {list(df.columns)}. Use Retrosheet gameinfo.csv, not the older MLB API data.')
    raw = len(df)
    # Retrosheet encodes dates as YYYYMMDD integers/strings, not Unix timestamps.
    # Parse that format explicitly; also accept ISO dates for compatible extracts.
    dates_raw = df['date'].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    numeric_ymd = dates_raw.str.fullmatch(r'\d{8}').fillna(False)
    parsed_dates = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    parsed_dates.loc[numeric_ymd] = pd.to_datetime(dates_raw.loc[numeric_ymd], format='%Y%m%d', errors='coerce')
    parsed_dates.loc[~numeric_ymd] = pd.to_datetime(dates_raw.loc[~numeric_ymd], errors='coerce')
    df['date'] = parsed_dates
    if df['date'].notna().sum() == 0:
        raise ValueError('Could not parse any dates. Example raw values: ' + str(dates_raw.head(8).tolist()))
    df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')
    # Retrosheet documents the regular-season group separately; variants depend on extract.
    kinds = df['gametype'].astype(str).str.strip().str.lower()
    observed = sorted(kinds.unique().tolist())
    allowed = {'regular', 'regular season', 'regular-season', 'reg', 'r'}
    regular = kinds.isin(allowed)
    if not regular.any():
        raise ValueError(f'No regular-season games found. Available gametype values: {observed}. Check encoding before changing the filter.')
    df = df.loc[regular].copy()
    after_reg = len(df)
    df = df[df['date'].notna() & df['hometeam'].notna() & df['visteam'].notna()].copy()
    df['year'] = df['date'].dt.year
    df = df[df['year'].between(2008, 2024) & ~df['year'].isin(EXCLUDED_YEARS)].copy()
    before_att = len(df)
    df = df[df['attendance'].notna() & (df['attendance'] > 0)].copy()
    # Fail rather than silently count duplicate game records twice.
    dup = int(df['gid'].duplicated().sum())
    if dup:
        raise ValueError(f'Found {dup} duplicate gid values after filtering. Inspect records before dropping any.')
    df['month'] = df['date'].dt.month.astype(str)
    df['weekday'] = df['date'].dt.dayofweek.astype(str)
    df = df.sort_values(['date', 'gid']).reset_index(drop=True)
    counts = {
        'rows_raw': int(raw), 'rows_regular_before_other_filters': int(after_reg),
        'rows_2008_2024_nonpandemic_before_attendance_filter': int(before_att),
        'rows_excluded_invalid_attendance': int(before_att-len(df)),
        'rows_usable': int(len(df)), 'year_counts': {str(k):int(v) for k,v in df.groupby('year').size().items()},
        'gametype_values_in_source': observed,
    }
    if not ((df['year'] == 2023).any() and (df['year'] == 2024).any()):
        raise ValueError('Missing games from 2023 or 2024 after filtering. Counts by year: ' + str(counts['year_counts']) + '; original gametype values: ' + str(observed))
    return df, counts


def date_bootstrap(y, benchmark, ridge_pred, dates, reps=2000):
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame({'date': dates, 'base_error': np.abs(y-benchmark), 'ridge_error': np.abs(y-ridge_pred)})
    grouped = frame.groupby('date', sort=False).agg(base_sum=('base_error','sum'), ridge_sum=('ridge_error','sum'), n=('date','size'))
    base_sum = grouped['base_sum'].to_numpy()
    ridge_sum = grouped['ridge_sum'].to_numpy()
    counts = grouped['n'].to_numpy()
    b = len(grouped)
    values = np.empty((reps,3))
    for i in range(reps):
        indexes = rng.integers(0, b, size=b)
        total = counts[indexes].sum()
        bm = base_sum[indexes].sum()/total
        rm = ridge_sum[indexes].sum()/total
        values[i] = [bm, rm, bm-rm]
    return {
        'benchmark_mae_95ci': np.percentile(values[:,0],[2.5,97.5]).round(3).tolist(),
        'ridge_mae_95ci': np.percentile(values[:,1],[2.5,97.5]).round(3).tolist(),
        'improvement_benchmark_minus_ridge_95ci': np.percentile(values[:,2],[2.5,97.5]).round(3).tolist(),
        'bootstrap_reps': reps, 'date_blocks': int(b), 'seed': SEED,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gameinfo', required=True, help='Path to the downloaded Retrosheet gameinfo.csv')
    p.add_argument('--output', default='outputs')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df, counts = load_data(args.gameinfo)

    # Forward-chaining tuning on pre-2024 seasons. 2024 is validation, never used to tune alpha.
    folds = [y for y in [2016,2017,2018,2019,2022,2023] if (df['year']==y).any() and (df['year']<y).sum()>0]
    fold_records = []
    for alpha in ALPHAS:
        maes = []
        for yr in folds:
            tr = df[df['year'] < yr]
            va = df[df['year'] == yr]
            model = make_model(alpha)
            model.fit(tr[CATEGORICAL],tr['attendance'])
            mae = mean_absolute_error(va['attendance'],model.predict(va[CATEGORICAL]))
            maes.append(float(mae))
        fold_records.append({'alpha':float(alpha),'mean_fold_mae':float(np.mean(maes)),'fold_maes':dict(zip(map(str,folds),maes))})
    best = min(fold_records,key=lambda r:r['mean_fold_mae'])
    train = df[df['year'] <= 2023]
    val = df[df['year'] == 2024].copy()
    mdl = make_model(best['alpha'])
    mdl.fit(train[CATEGORICAL],train['attendance'])
    ridge_pred = mdl.predict(val[CATEGORICAL])
    prior = df[df['year']==2023].groupby('hometeam')['attendance'].mean()
    fallback = float(df[df['year']==2023]['attendance'].mean())
    missing_teams = sorted(set(val['hometeam'])-set(prior.index))
    benchmark_pred = val['hometeam'].map(prior).fillna(fallback).to_numpy()
    y = val['attendance'].to_numpy()
    bs = date_bootstrap(y,benchmark_pred,ridge_pred,val['date'].to_numpy())
    out = {
        'source':'Retrosheet gameinfo.csv', 'data_url':'https://www.retrosheet.org/downloads/csvdownloads.html',
        'filter':'regular-season, 2008-2024, excluding 2020 and 2021; positive observed attendance',
        'data_counts':counts,
        'split':{'train_years':'2008-2023 excluding 2020, 2021','validation_year':2024,'final_test_year':2025,'final_test_used':False,'tuning_folds':folds},
        'inputs':CATEGORICAL, 'raw_input_count':len(CATEGORICAL),
        'ridge_alpha_grid':list(map(float,ALPHAS)), 'alpha_cv':fold_records, 'best_alpha':best['alpha'],
        'historical_benchmark':'2023 mean attendance by home team; missing teams use 2023 league mean',
        'benchmark_unknown_home_teams_2024':missing_teams,
        'validation_n':len(val),
        'historical_benchmark_mae':float(mean_absolute_error(y,benchmark_pred)),
        'ridge_mae':float(mean_absolute_error(y,ridge_pred)),
        'ridge_rmse':float(np.sqrt(mean_squared_error(y,ridge_pred))),
        'ridge_r2':float(r2_score(y,ridge_pred)),
        'improvement_benchmark_minus_ridge':float(mean_absolute_error(y,benchmark_pred)-mean_absolute_error(y,ridge_pred)),
        'uncertainty':bs,
    }
    (output/'baseline_results.json').write_text(json.dumps(out,indent=2))
    pd.DataFrame({'gid':val['gid'],'date':val['date'].dt.strftime('%Y-%m-%d'),'home_team':val['hometeam'],
                  'attendance':y,'benchmark_prediction':benchmark_pred,'ridge_prediction':ridge_pred}).to_csv(output/'validation_predictions_2024.csv',index=False)
    md = f'''# MLB Attendance Prediction: Baseline Checkpoint\n\n## Group\nIrenka, Kate, Rachel — Canvas group: [insert]\n\n## Changes since the proposal\nWe switched to Retrosheet's public bulk CSV download, as recommended, and restricted the primary analysis to regular-season games. We exclude 2020 and 2021 because pandemic restrictions disrupted attendance. We added the previous-season home-team attendance benchmark.\n\n## Data\nSource: [Retrosheet CSV downloads](https://www.retrosheet.org/downloads/csvdownloads.html), `gameinfo.csv`. The response is attendance (number of attendees); the four raw inputs are home team, visiting team, month and weekday. We used {counts['rows_usable']:,} games after filtering 2008–2024 regular-season games, excluding 2020–2021, missing/nonpositive attendance, and invalid dates or teams. The source contained {counts['rows_raw']:,} game records; {counts['rows_excluded_invalid_attendance']:,} eligible-year records lacked usable attendance. Full annual counts are in `baseline_results.json`. Weather is deferred; team performance will use prior games only.\n\n## Split\nTraining: eligible games through 2023. Forward-chaining tuning folds: {', '.join(map(str,folds))}, fitting only on earlier eligible years each time. Validation: {len(val):,} games in 2024. The 2025 test set remains untouched. Bootstrap RNG seed: 42.\n\n## Baseline\nOne-hot encoded home team, visiting team, month and weekday, followed by ridge regression. We selected α from 13 log-spaced values between 0.001 and 1000 using average forward-chaining validation MAE. Selected α = {best['alpha']:.4g}. The floor is the 2023 mean attendance of each home team (with 2023 overall mean as fallback for unseen teams).\n\n## Result\nOn 2024, the historical benchmark MAE was {out['historical_benchmark_mae']:,.2f} attendees and ridge MAE was {out['ridge_mae']:,.2f} attendees (RMSE {out['ridge_rmse']:,.2f}, R² {out['ridge_r2']:.4f}). The paired MAE improvement (benchmark minus ridge; positive means ridge wins) was {out['improvement_benchmark_minus_ridge']:,.2f} attendees, with a 95% date-block bootstrap interval of {bs['improvement_benchmark_minus_ridge_95ci']} attendees. Ridge's own MAE 95% interval was {bs['ridge_mae_95ci']}. We resampled entire 2024 game dates 2,000 times to preserve within-date dependence. If the paired interval includes zero, we treat the models as tied.\n\n## Issues found\nWe avoided current-game score/performance features that would leak future information; the contextual baseline uses only pregame identifiers and schedule information. Retrosheet weather fields exist but have not yet been assessed for completeness or whether they are known before attendance is predicted. Pandemic seasons were excluded. {('The 2024 data also include home teams absent in 2023, for which we used the overall 2023 mean: ' + ', '.join(missing_teams) + '.') if missing_teams else 'Every 2024 home team had a corresponding 2023 home-team historical mean.'}\n\n## Next steps\nKate: engineer strictly pregame team win-rate/scoring features from Retrosheet `teamstats.csv`, handling early-season records with prior-year summaries. Irenka: compare contextual ridge with ridge plus past-performance features, then evaluate tree methods on identical splits. Rachel: evaluate paired MAE intervals and develop the central comparison figure. Everyone: check leakage and draft the paper.\n\n## Code\n[Insert link to GitHub repository containing `run_baseline.py`, requirements, Retrosheet source link, and the result files.]\n\nData attribution: The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.\n'''
    (output/'baseline_checkpoint_generated.md').write_text(md)
    print(json.dumps({'output_folder':str(output),'n_2024':len(val),'best_alpha':best['alpha'],'benchmark_mae':out['historical_benchmark_mae'],'ridge_mae':out['ridge_mae']},indent=2))

if __name__ == '__main__':
    main()
