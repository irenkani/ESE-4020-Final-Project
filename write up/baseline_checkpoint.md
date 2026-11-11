# MLB Attendance Prediction: Baseline Checkpoint

## Group
Irenka, Kate, Rachel — Canvas group: 4020 touse

## Changes since the proposal
Following the proposal feedback, we adopted Retrosheet's publicly downloadable game-level CSV data instead of relying on individual MLB API requests. We restricted the analysis to regular-season games, excluded 2020 (games without fans) and 2021 (attendance restrictions), and deferred weather until its availability and timing can be verified. We also added a required simple benchmark: the home team's mean attendance from the previous season. Our baseline checkpoint focuses on this benchmark and a contextual ridge model before testing the contribution of recent team performance.

## Data
Our source is Retrosheet's public [CSV downloads](https://www.retrosheet.org/downloads/csvdownloads.html), specifically `gameinfo.csv`. Each row represents a game, with attendance (number of spectators) as the prediction target. We use four raw input variables: home team, visiting team, month, and weekday; the categorical inputs are one-hot encoded for modeling. The source file contained 226,221 game records across its full historical coverage. After selecting 2008–2024 regular-season games, excluding 2020–2021, and removing records with unusable dates, team identifiers, or missing/nonpositive attendance, we retained 36,270 games. Among the 36,439 otherwise eligible records, 169 were excluded for invalid or missing attendance. The validation set contains 2,413 games from 2024.

## Split
We split by season, never randomly across games. Training uses eligible seasons through 2023. Ridge hyperparameters were selected using forward-chaining validation folds in 2016, 2017, 2018, 2019, 2022, and 2023, with each fold fitted only on earlier eligible seasons. We evaluated the selected model on 2024 once as the checkpoint validation year and reserved 2025 for the final test, untouched during model selection. The bootstrap random seed was 42.

## Baseline
The simplest benchmark predicts attendance at each 2024 game using that home team's average 2023 home-game attendance (falling back to the 2023 overall average if the team is unseen). The course baseline is ridge regression on one-hot encoded home team, visiting team, month, and weekday. We tuned its regularization parameter over 13 log-spaced values from $10^{-3}$ to $10^3$, selecting the value with the lowest mean MAE across forward-chaining folds. The selected penalty was $\alpha = 3.1623$.

## Result
On 2,413 held-out 2024 games, the previous-season home-team attendance benchmark achieved MAE **5,308.39** spectators. Ridge regression achieved MAE **5,893.94** spectators, RMSE **7,270.12**, and $R^2 = 0.5411$. Its MAE 95% confidence interval was **[5,703.46, 6,110.96]** spectators. Defining improvement as benchmark MAE minus ridge MAE (positive values mean ridge is better), the paired difference was **−585.55** spectators, with a 95% confidence interval of **[−836.88, −331.90]**. We calculated these intervals using 2,000 bootstrap resamples of entire game dates (185 distinct dates), keeping games on the same date together. Because the paired interval excludes zero, the historical benchmark outperformed this contextual ridge model on 2024 validation data; the two methods are not tied under this procedure.

## Issues found
The historical attendance benchmark was stronger than our contextual ridge baseline, indicating that the four categorical inputs alone may not capture year-to-year differences in home-team attendance. We deliberately excluded current-game scores and other postgame statistics to avoid leakage. Pandemic-era attendance is not representative of typical games, motivating our exclusion of 2020–2021. We have not yet verified whether the available weather fields are complete or consistently known at the intended prediction time, so they are omitted. Some 2024 games also have unusually large errors, which we will inspect before expanding the models; we will not use these cases to tune the existing 2024 validation result.

## Next steps
**Kate** will construct recent win percentage and scoring statistics from *prior* games only, with documented handling for the first games of each season. **Irenka** will compare ridge with and without these historical performance features and then test tree-based methods using identical chronological splits. **Rachel** will calculate paired bootstrap confidence intervals for those model comparisons and develop the central figure. We will jointly audit leakage, document missingness, and draft the early paper. The final research question is whether recent team performance reduces held-out MAE beyond contextual features by more than the uncertainty in the paired difference.

## Code
[Add GitHub repository URL containing `run_baseline.py`, `requirements.txt`, the data source instructions, `baseline_results.json`, and `validation_predictions_2024.csv`.]

Data attribution: The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.
