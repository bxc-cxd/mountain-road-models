# Segment-panel analysis

Python 3.12:

```bash
python -m pip install -r requirements.txt
python run.py --input data.xlsx --output results --workers 4
```

The input workbook contains three sheets: `路段基础数据`, `月度面板数据` and
`指标计算输入`. Column names and units are in `data dictionary.csv`.
`examples/simulated panel.xlsx` provides the input format with simulated values.
Monthly records must form a balanced panel with treated and control segments.
Economic calculations use the first 12 effective months; speed comparisons also
require 12 months before construction for segments with speed-management measures.

The workflow calculates indicators and baseline covariates, propensity weights,
GEE and fixed-effects models, event contrasts, bootstrap intervals, cohort-time
estimates, economic rankings, valuation scenarios and indicator sensitivity.
The default bootstrap count is 2,000. Results are saved as CSV, JSON and NumPy files.

```bash
python run.py --input "examples/simulated panel.xlsx" --output "results example" --bootstrap-reps 20
```

`--steps` selects analyses and includes their dependencies. Available stages are
`models`, `event study`, `comparisons`, `bootstrap`, `staggered`, `economics`,
`monetary` and `indicator sensitivity`. `--verbose` displays stage progress.

`--indicator-inputs inputs.csv` can supply the indicator inputs instead of the
third sheet. Its columns are `segment`, `aadt`, `length_used`, `width_used`,
`low_road_class` and `measure_count`. Segment order follows the base sheet.

`--exclude-year` selects the omitted year; the default is the first effective
intervention year. `--reference-year` selects the candidate evaluation year;
the default is the last complete control-group year. `--valuation parameters.json`
supplies unit costs, CPI values and price years; defaults are in
`parameters/valuation.json`.
