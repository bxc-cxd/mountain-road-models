# Mountain road micro-retrofit models

## Run

Use Python 3.12 with the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python run.py --input /path/to/study.xlsx --output /path/to/results --workers 4
```

The default is 2,000 bootstrap replicates. To run the supplied simulated example:

```bash
python run.py --input examples/simulated_panel.xlsx --output results_demo --mode demo --bootstrap-reps 20 --workers 2
```

## Inputs and analysis

The workbook contains `路段基础数据` and `月度面板数据`. Required fields and units
are listed in `data_dictionary.csv`. `examples/simulated_panel.xlsx` contains
simulated data and can be generated with `examples/generate_demo.py`.

`analysis/indicators.py` defines the two indicators. The baseline risk index is
read from the base sheet, and monthly maintenance scores are calculated before
baseline aggregation. The fixed normal sequence follows base-sheet segment
order and chronological calendar months.

`run.py` executes the following stages:

| Stage | Analysis |
|---|---|
| `models` | Baseline covariates, propensity scores, ATT and overlap weights, NB and Poisson GEE, sample and dispersion comparisons, matched before-after and SPF/EB estimates |
| `event_study` | Event-window contrasts and exclusion of construction months |
| `legacy_comparisons` | Segment and calendar-month fixed effects, placebo timing, high-crash exclusion, cohort interactions and risk interactions |
| `bootstrap` | Segment resampling with propensity, dispersion and outcome models refitted |
| `staggered` | Cohort-time Poisson imputation and segment bootstrap |
| `economics` | Crashes prevented, cost-effectiveness, prioritization, uncertainty and budget scenarios |
| `monetary` | Crash-cost valuation and conversion to a common price year |

Use `--steps` to select stages whose prerequisite outputs are available.
Results are written to `models`, `event`, `economics` and `monetary` under the
output directory. Model inputs and design matrices are stored in `cache`.

## Indicator sensitivity

The indicator sensitivity analysis varies eight coefficients individually by
plus or minus 20%. Each scenario refits propensity weights, dispersion and GEE.
The construction-input CSV contains `segment`, `aadt`, `length_used`,
`width_used`, `low_road_class` and `measure_count` in their original units.

```bash
python run.py --input /path/to/study.xlsx --output /path/to/results --steps indicator_sensitivity --indicator-inputs /path/to/indicator_inputs.csv
```
