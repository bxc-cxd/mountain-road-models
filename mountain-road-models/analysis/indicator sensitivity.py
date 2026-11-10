import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from . import models
from .config import INPUT, folder
from .inputs import indicator_inputs, risk_arguments, indicator_digest
from .indicators import calculate_risk

def main():
    b, _, _, _, _, _ = models.prepare()
    supplied = indicator_inputs(INPUT, b.segment, required=True)
    original_hash = indicator_digest(supplied)
    values = risk_arguments(supplied)
    defaults = {'aadt_weight': 0.38, 'length_weight': 0.33, 'inverse_width_weight': 0.2, 'low_class_weight': 0.16, 'measure_count_weight': 0.12, 'risk_noise_sd': 0.45, 'score_risk_slope': -4.5, 'score_noise_sd': 5.0}
    weights = list(defaults)[:5]
    scenarios = [('baseline', defaults.copy())]
    for name in defaults:
        for factor, label in [(0.8, 'minus20'), (1.2, 'plus20')]:
            params = defaults.copy()
            params[name] *= factor
            scenarios.append((name + '_' + label, params))
    out = folder('indicator sensitivity')
    registry = [dict(scenario=name, **params) for name, params in scenarios]
    (out / 'scenario registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
    rows = []
    for name, params in scenarios:
        risk = calculate_risk(*values, weights=[params[k] for k in weights], noise_scale=params['risk_noise_sd'])
        _, _, p, _, audit, _ = models.prepare(risk_values=risk, maintenance_parameters={'risk_slope': params['score_risk_slope'], 'noise_scale': params['score_noise_sd']})
        alpha, _ = models.estimate_alpha(p)
        for weighting in ['att_weight', 'overlap_weight']:
            fit = models.fit_gee(p, alpha, weighting)
            row = models.contrast(fit, p, name, weighting)
            row.update(weighting=weighting, alpha=alpha)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'effect estimates.csv', index=False)
    estimates = pd.DataFrame(rows)
    summary = {}
    for weighting in ['att_weight', 'overlap_weight']:
        selected = estimates.loc[estimates.weighting.eq(weighting) & estimates.analysis.ne('baseline')]
        summary[weighting] = {'scenarios': len(selected), 'irr_min': float(selected.irr.min()), 'irr_max': float(selected.irr.max()), 'reduction_percent_min': float(100 * (1 - selected.irr.max())), 'reduction_percent_max': float(100 * (1 - selected.irr.min()))}
    summary.update(input_sha256=audit['input_sha256'], construction_inputs_sha256=original_hash)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
