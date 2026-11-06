# Mountain road micro-retrofit models

Analysis code for the Fujian micro-retrofit study, including propensity weighting,
count models, event studies, segment bootstraps and economic evaluation.

## Data and installation

The government study data are not distributed. The supplied
`examples/simulated_panel.xlsx` is an independently simulated example. Its results
demonstrate the workflow and are not the manuscript estimates.

Inputs have two sheets, `路段基础数据` and `月度面板数据`. Column names, units and roles
are listed in `data_dictionary.csv`, encoded as UTF-8 with a byte-order mark.

Use Python 3.12 and install the pinned dependencies in a virtual environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate `.venv` before the installation command.

## Run the analysis

Run commands from the repository root. The supplied example uses 20 bootstrap
replicates for an execution check:

```bash
python run.py --input examples/simulated_panel.xlsx --output results_demo --mode demo --bootstrap-reps 20 --workers 2
```

The example can also be generated from its fixed seed:

```bash
python examples/generate_demo.py --output data/demo.xlsx
```

For an authorized local study workbook, the default is 2,000 bootstrap replicates:

```bash
python run.py --input /path/to/study.xlsx --output /path/to/local_results --mode study --workers 4
```

Input workbooks are read without modification and checked by SHA-256 after
execution. Study mode also checks the documented sample totals. Outputs include
segment records and model caches, so outputs from restricted data should remain
local. The workflow runs these stages in dependency order:

| Stage | Analyses |
|---|---|
| `models` | Propensity scores, ATT and overlap weights, NB and Poisson GEE, common-sample comparisons, exclusion of 2022, dispersion sensitivity, matched before-after and SPF/EB analyses |
| `event_study` | Event-window contrasts, construction-period exclusion and calendar-aligned summaries |
| `legacy_comparisons` | Segment/month fixed effects, placebo timing, high-crash exclusion, cohort-interacted comparison and risk interactions |
| `bootstrap` | Stratified segment resampling with propensity, dispersion, outcome and averaging-horizon refitting |
| `staggered` | Cohort-time Poisson imputation with segment bootstrap |
| `economics` | First-year benefits, cost-effectiveness, priorities, uncertainty, budget scenarios, candidate screening and speed comparisons |
| `monetary` | Literature-based crash-cost bounds at 2024 prices |

Use `--steps` to rerun stages after their required outputs exist, for example:

```bash
python run.py --input /path/to/study.xlsx --output /path/to/local_results --steps economics monetary
```

Bootstrap failures and boundary fits are recorded. Failed draws are not replaced.

## Analysis settings

The common propensity-score baseline ends before the earliest effective
intervention month. Covariates are transformed pre-period crash rate, log AADT,
log segment length, lane width, baseline speed, baseline maintenance score and
baseline risk index. Controls receive no pseudo-treatment date.

The main model uses normalized ATT weights, negative-binomial GEE, an independence
working correlation, segment-clustered robust covariance, calendar-month indicators
and a log vehicle-kilometre offset. Predictors include treatment group, an immediate
treatment term and a post-intervention trend. The average effect exponentiates the
level-and-trend contrast at the weighted mean observed treated post month; its
standard error uses the covariance of both coefficients. The full bootstrap
re-estimates this procedure. Overlap weighting is used for comparison and risk
interactions.

Economic simulations draw the level and trend jointly 5,000 times while holding
the remaining fitted components fixed. Candidate screening excludes costs and
proposed measures for untreated controls. Monetary comparisons hold the
preconstruction severity mix fixed and convert construction-year engineering
costs to 2024 prices. Crash costs are from Tan et al. (2020), *Cost analysis of
road traffic crashes in China*, https://doi.org/10.1080/17457300.2020.1785507.
Price conversion uses the National Bureau of Statistics annual all-items CPI
with the previous year equal to 100.

## Score formulas

The analysis reads the risk and maintenance columns supplied in the workbook.
`analysis/indicators.py` implements the score formulas separately, including
their fixed pseudo-random terms and seed. Risk construction requires the original
construction length and width, which may differ from the checked geometry used
to calculate exposure.

The maintenance formula has an upper bound of 100. It does not recalculate the
existing workbook scores used in the reference analysis. Using recalculated
scores requires a new analysis run. To export a comparison:

```bash
python examples/recalculate_maintenance.py --input /path/to/study.xlsx --output /path/to/local_scores.csv
```

Workbook row order determines segment order in the fixed random sequence.

## Verification and outputs

```bash
python verify.py
```

This checks the published file hashes, Python syntax, local module paths and data
dictionary. `VALIDATION.md` describes numerical comparisons and the checks on this
release. `validation_summary.json` records their results.

| Content | Output |
|---|---|
| Sample and timing | `models/data_audit.json`, `event/input_audit.json` |
| Balance | `models/balance.csv`, `models/balance_metadata.json` |
| Calendar-aligned summaries | `event/table4_calendar_aligned.csv` |
| Model estimates | `models/model_results.csv` |
| Full bootstrap | `models/bootstrap_summary.json` |
| Cohort-time imputation | `models/cohort_time_metadata.json` |
| FE/placebo/high-crash comparisons | `models/legacy_comparisons.csv` |
| Risk interactions | `models/risk_interaction_results.csv`, `models/risk_interaction_metadata.json` |
| Event windows | `event/event_results.csv` |
| Economic estimates | `economics/economic_summary.csv` |
| Implemented-segment priorities | `economics/treated_ranking_12months_three.csv` |
| Budget scenarios | `economics/budget_scenarios.csv` |
| Candidate screening | `economics/candidate_screening.csv` |
| Speed comparison | `economics/speed_summary.json` |
| Monetary bounds | `monetary/monetary_results.csv` |
