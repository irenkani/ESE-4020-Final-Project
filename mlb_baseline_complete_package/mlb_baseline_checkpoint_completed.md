# Baseline Checkpoint: Does Recent Team Performance Improve MLB Attendance Predictions?

## Group
Irenka, Kate, Rachel (Canvas group number: [add group number])

## Changes since the proposal
We narrowed the first experiment to a reproducible contextual attendance baseline, before adding recent team performance features. Following the feedback, we added the previous-season home-team mean attendance as the simple benchmark that all later models should beat. We excluded 2020 and 2021 from the main fitting/evaluation sample because COVID-era attendance restrictions make those seasons difficult to compare with normal seasons. We also removed same-game performance statistics from the baseline so that predictions use only information available before the game. We used an existing locally saved MLB game-level CSV rather than downloading individual games during this checkpoint.

## Data
We loaded `attendance_mlbdata.csv`, an existing file originally assembled using MLB game information. It has 50,237 rows and contains game IDs, dates, game type/status, home and away teams, attendance, and contextual fields. We selected completed regular-season games (`game_type == "R"` and status `Final` or `Completed Early`), removed 34 extra records with repeated game IDs (retaining the latest completion-date record), and removed 1,048 games with missing or nonpositive attendance. After excluding 2020 and 2021, we had 38,413 eligible games from 2008–2025. We set aside all 2,416 eligible 2025 games untouched, leaving **33,597 training games through 2023** and **2,400 games for 2024 validation**.

The outcome is attendance (attendees per game). The baseline has **four raw input fields**: home team, away team, month, and day of week, expanded to **79 one-hot-encoded predictors** when fit on the full training period. Although the file contains observed weather, we omitted it from this initial baseline; weather reported after a game could be unavailable at prediction time.

Source-link issue: the uploaded CSV is usable locally, but we still need to provide a **public, directly downloadable copy** or document a compliant alternative, because an API documentation page alone does not establish that the dataset is an existing public download.

## Split
We used chronological training/validation, not a random split. Training contains eligible 2008–2023 games, excluding 2020–2021, and validation is 2024. For hyperparameter selection, we used forward-chaining annual folds: train on years strictly before the validation year, then validate on **2015, 2016, 2017, 2018, 2019, 2022, and 2023** in turn. The **2025 season remains untouched** for final testing. We used **random seed 42** for bootstrap resampling; the ridge fit itself is deterministic.

## Baseline
The simple historical benchmark predicts each 2024 game's attendance using **that home team's mean attendance in 2023**. Our course baseline is ridge regression with one-hot encoding of home team, away team, month, and weekday. The encoder is fit only on each fold's training data. We tested 13 logarithmically spaced ridge penalties, **alpha = 0.001 through 1,000**, selecting the penalty with the lowest mean forward-chaining validation MAE. The chosen penalty was **alpha = 10** (mean forward-chaining CV MAE: **5,917.19** attendees).

## Result
On the 2,400 games in 2024, the **historical benchmark** achieved **MAE = 5,308.03**, **RMSE = 6,946.79**, and **R² = 0.581**. The **ridge baseline** achieved **MAE = 5,890.70**, **RMSE = 7,258.84**, and **R² = 0.542**.

We used **2,000 date-block bootstrap replicates** with seed 42, sampling 2024 game dates with replacement and keeping all games from the same date together. The ridge MAE 95% bootstrap interval was **[5,703.29, 6,090.71]**, while the historical benchmark's interval was **[5,077.36, 5,549.76]**. The *paired* improvement from ridge over the benchmark (historical MAE minus ridge MAE) was **−582.67 attendees**, with **95% CI [−844.01, −341.24]**. Because the interval lies entirely below zero, ridge is meaningfully worse on this validation set rather than statistically tied. The simple previous-season team average is therefore a strong benchmark for later models to beat.

## Issues found
The uploaded attendance file mixes regular-season games with spring training and postseason entries, so game-type filtering was essential. It also repeats some game IDs for postponed or resumed games, creating a risk of duplicated target observations; we kept one record per game ID. We removed missing attendance observations and pandemic seasons. The previous notebook also used a random train/test split and same-game total runs, which could lead to invalid pregame prediction results; neither was used in this baseline. Finally, the current CSV does not yet have a verified public download URL, which is a remaining compliance issue.

## Next steps
- **Kate:** Verify and document a public downloadable data source, finalize data-quality checks, and construct recent team performance features using **only games before each target date** (with previous-season history as an early-season fallback).
- **Irenka:** Add the recent performance predictors to ridge and compare the expanded model against this baseline; then implement tree-based models using the same chronological evaluation.
- **Rachel:** Produce side-by-side held-out MAE figures and date-block bootstrap intervals for the *paired improvement* from adding performance, and document uncertainty and limitations.

Together, we will interpret whether recent performance lowers validation MAE by more than its error bar and prepare the early paper draft.

## Code
`mlb_baseline_actual.py` — reproducible script producing `mlb_baseline_results.json` and `mlb_2024_validation_predictions.csv` from the uploaded `attendance_mlbdata.csv`. Add a public GitHub or notebook URL here before submission: [link].
