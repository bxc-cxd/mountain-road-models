# Validation

## Package and output checks

```bash
python verify.py
python verify.py --results /path/to/current_results --reference /path/to/reference_results --report comparison.json
```

The first command verifies file hashes, syntax, local module paths and the data
dictionary. The second also compares 21 CSV outputs and selected fields in four
JSON outputs. CSV comparisons require identical columns and row counts, equal
non-numeric values, and numerical agreement within relative and absolute
tolerances of 1e-8. JSON fields use tolerances of 1e-9. Elapsed times and local
paths are excluded. Any failed check returns a nonzero exit status.

Reference comparisons require the same input workbook, analysis settings and
bootstrap count. The study reference outputs are held locally. The release
checks and numerical comparison results are recorded in `validation_summary.json`.

## Previous study validation

On 2026-11-05, version 0.1.0 was run against the study workbook. Its 25 numerical
outputs matched the archived revision analyses. Both study bootstraps completed
2,000 valid replicates. These are checks of the supplied workbook scores, as
described in the README; they are not results from recalculated score inputs.

## Version 0.1.1

On 2026-11-06, the model archive was extracted into a separate directory and all
seven analysis stages ran on the supplied simulated workbook. Both segment
bootstraps completed 20 valid replicates. The 25 output comparisons matched the
previous simulated-data run within the tolerances above. Four command-line
entry points and score recalculation were also checked. Source comparison
confirmed that the statistical calculations were unchanged after removal of
the plotting code. The study models were not rerun for this packaging change.
