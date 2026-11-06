from pathlib import Path
import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import tokenize

CSV_OUTPUTS = [
    'models/model_results.csv',
    'models/primary_parameters.csv',
    'models/primary_covariance.csv',
    'models/balance.csv',
    'models/bootstrap_draws.csv',
    'models/cohort_time_effects.csv',
    'models/cohort_time_bootstrap.csv',
    'models/spf_eb_segments.csv',
    'event/event_results.csv',
    'event/table4_calendar_aligned.csv',
    'event/figure4_unsmoothed_rates.csv',
    'economics/economic_summary.csv',
    'economics/treated_ranking_12months_three.csv',
    'economics/treated_ranking_12months_risk_ce.csv',
    'economics/treated_ranking_12months_ce.csv',
    'economics/treated_ranking_observed_followup.csv',
    'economics/budget_scenarios.csv',
    'economics/effect_assumption_scenarios.csv',
    'economics/speed_segment_changes.csv',
    'economics/candidate_screening.csv',
    'monetary/monetary_results.csv',
]
JSON_FIELDS = {
    'models/bootstrap_summary.json': ['point_estimate', 'ci_low', 'ci_high', 'replicates_valid'],
    'models/cohort_time_metadata.json': ['ratio', 'ci_low', 'ci_high', 'bootstrap_valid'],
    'models/before_after.json': ['irr', 'ci_low', 'ci_high'],
    'models/spf_eb.json': ['estimate', 'ci_low', 'ci_high'],
}


def check_package(root):
    checked = []
    modules = []
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Invalid manifest path: {name}')
        if not path.is_file():
            raise ValueError(f'Missing package file: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        checked.append(name)
        if path.suffix == '.py':
            source = path.read_text(encoding='utf-8-sig')
            compile(source, name, 'exec')
            tree = ast.parse(source)
            if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)):
                raise ValueError(f'Unexpected Python comment: {name}')
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node, clean=False) is not None:
                        raise ValueError(f'Unexpected Python docstring: {name}')
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    targets = [node.module.split('.')[0]] if node.module else [alias.name for alias in node.names]
                    for target in targets:
                        if not (path.parent / (target + '.py')).is_file() and not (path.parent / target / '__init__.py').is_file():
                            raise ValueError(f'Missing local module: {name} -> {target}')
            modules.append(name)
    with (root / 'data_dictionary.csv').open(encoding='utf-8-sig', newline='') as stream:
        fields = list(csv.DictReader(stream))
    if not any(row['sheet'] == '月度面板数据' and row['column'] == '月份' and row['unit_or_type'] == '1 to 12' for row in fields):
        raise ValueError('Invalid calendar-month range in data_dictionary.csv')
    return {'files_checked': len(checked), 'python_files_checked': len(modules), 'passed': True}


def compare_outputs(results, reference):
    import numpy as np
    import pandas as pd

    checks = []
    for name in CSV_OUTPUTS:
        actual = pd.read_csv(results / name)
        expected = pd.read_csv(reference / name)
        if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
            raise ValueError(f'Output structure differs: {name}')
        largest = 0.0
        for column in actual.columns:
            if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(expected[column]):
                x = actual[column].to_numpy(float)
                y = expected[column].to_numpy(float)
                if not np.allclose(x, y, rtol=1e-8, atol=1e-8, equal_nan=True):
                    raise ValueError(f'Numerical mismatch: {name}, {column}')
                delta = np.abs(x - y)
                finite = delta[np.isfinite(delta)]
                if finite.size:
                    largest = max(largest, float(finite.max()))
            elif not actual[column].fillna('').astype(str).equals(expected[column].fillna('').astype(str)):
                raise ValueError(f'Value mismatch: {name}, {column}')
        checks.append({'output': name, 'rows': len(actual), 'max_numeric_abs_difference': largest, 'passed': True})
    for name, keys in JSON_FIELDS.items():
        actual = json.loads((results / name).read_text(encoding='utf-8'))
        expected = json.loads((reference / name).read_text(encoding='utf-8'))
        for key in keys:
            if not np.isclose(actual[key], expected[key], rtol=1e-9, atol=1e-9):
                raise ValueError(f'Numerical mismatch: {name}, {key}')
        checks.append({'output': name, 'fields': keys, 'passed': True})
    return checks


def main():
    parser = argparse.ArgumentParser(description='Verify package files and optionally compare two analysis runs.')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if (args.results is None) != (args.reference is None):
        parser.error('--results and --reference must be provided together.')
    result = {'package': check_package(Path(__file__).resolve().parent)}
    if args.results is not None:
        result['output_comparisons'] = compare_outputs(args.results, args.reference)
        result['outputs_compared'] = len(result['output_comparisons'])
    result['passed'] = True
    text = json.dumps(result, indent=2)
    if args.report is not None:
        if args.report.exists():
            parser.error('Report already exists. Choose a new output path.')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, SyntaxError) as error:
        print(f'Verification failed: {error}', file=sys.stderr)
        sys.exit(1)
