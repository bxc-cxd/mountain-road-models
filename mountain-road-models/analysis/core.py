from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
SEED = 20260721
from .config import BOOTSTRAP_REPS
from .indicators import maintenance_for_panel, calculate_risk
from .inputs import indicator_inputs, risk_arguments
SIMULATION_DRAWS = 5000

def weighted_stats(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    x = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = (x[keep], w[keep])
    if len(x) == 0 or w.sum() == 0:
        return (float('nan'), float('nan'))
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    return (mean, variance)

def smd(t_values: pd.Series, c_values: pd.Series, t_weights: pd.Series, c_weights: pd.Series) -> float:
    tm, tv = weighted_stats(t_values, t_weights)
    cm, cv = weighted_stats(c_values, c_weights)
    denom = math.sqrt((tv + cv) / 2) if np.isfinite(tv + cv) else float('nan')
    numerical_floor = 1e-10 * max(abs(tm), abs(cm), 1.0)
    if not np.isfinite(denom) or denom <= numerical_floor:
        return 0.0 if np.isclose(tm, cm, equal_nan=False) else float('nan')
    return (tm - cm) / denom

def normalize_road_class(value: object) -> str:
    text = str(value).strip()
    if text in {'2', '二级', '二级公路'}:
        return '二级'
    if text in {'三级', '三级公路'}:
        return '三级'
    if text in {'四级', '四级公路'}:
        return '四级'
    return text

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def normal_result_row(name: str, measure: str, result, term: str, scale: str='irr') -> dict:
    beta = float(result.params[term])
    se = float(result.bse[term])
    low, high = (beta - 1.959963984540054 * se, beta + 1.959963984540054 * se)
    if scale == 'irr':
        estimate, ci_low, ci_high = (math.exp(beta), math.exp(low), math.exp(high))
    else:
        estimate, ci_low, ci_high = (beta, low, high)
    return {'model': name, 'term': term, 'measure': measure, 'estimate': estimate, 'ci_low': ci_low, 'ci_high': ci_high, 'coefficient': beta, 'std_error': se, 'p_value': float(result.pvalues[term]), 'n_obs': int(result.nobs), 'converged': bool(getattr(result, 'mle_retvals', {}).get('converged', True))}

def fit_poisson(formula: str, data: pd.DataFrame, *, weights: pd.Series | None=None):
    model = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data['log_exposure'], freq_weights=weights)
    return model.fit(maxiter=500, method='irls', cov_type='cluster', cov_kwds={'groups': data['segment']})

def fit_nb2(formula: str, data: pd.DataFrame):
    model = smf.negativebinomial(formula, data=data, offset=data['log_exposure'])
    return model.fit(method='bfgs', maxiter=1000, disp=False, cov_type='cluster', cov_kwds={'groups': data['segment']})

def load_and_prepare(input_path: Path, *, risk_values=None, maintenance_parameters=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_raw = pd.read_excel(input_path, sheet_name='路段基础数据')
    panel_raw = pd.read_excel(input_path, sheet_name='月度面板数据')
    inputs = indicator_inputs(input_path, base_raw['路段编号'].astype(str), required=False)
    if risk_values is None:
        if inputs is not None:
            risk_values = calculate_risk(*risk_arguments(inputs))
        elif '基础风险指数' in base_raw:
            risk_values = pd.to_numeric(base_raw['基础风险指数']).to_numpy(float)
        else:
            raise ValueError('Missing indicator inputs or baseline risk values.')
    base = pd.DataFrame({'segment': base_raw['路段编号'].astype(str), 'route': base_raw['路线编号'].astype(str), 'road_class': base_raw['原始道路等级'].map(normalize_road_class), 'length_m': pd.to_numeric(base_raw['路段长度_米']), 'lanes': pd.to_numeric(base_raw['车道数']), 'lane_width': pd.to_numeric(base_raw['单车道宽度_米']), 'aadt': pd.to_numeric(base_raw['年平均日交通量']), 'treated': pd.to_numeric(base_raw['处理组标记']).astype(int), 'effective_date': pd.to_datetime(base_raw['干预生效月份'], errors='coerce'), 'cost_yuan': pd.to_numeric(base_raw['工程费用_元']).fillna(0), 'baseline_risk': np.asarray(risk_values, dtype=float), 'bundle_intensity': pd.to_numeric(base_raw['治理组合强度']).fillna(0)})
    panel = pd.DataFrame({'segment': panel_raw['路段编号'].astype(str), 'year': pd.to_numeric(panel_raw['年份']).astype(int), 'month': pd.to_numeric(panel_raw['月份']).astype(int), 'D': pd.to_numeric(panel_raw['干预后标记']).fillna(0).astype(int), 'time_after': pd.to_numeric(panel_raw['干预后月数']).fillna(0).astype(float), 'event_time': pd.to_numeric(panel_raw['事件时间_相对月份'], errors='coerce'), 'y': pd.to_numeric(panel_raw['月度总事故数_最终']).astype(int), 'exposure': pd.to_numeric(panel_raw['月度交通暴露量']).astype(float), 'speed': pd.to_numeric(panel_raw['第85百分位速度']).astype(float), 'rainfall': pd.to_numeric(panel_raw.get('降雨量_毫米', pd.Series(0, index=panel_raw.index))).astype(float), 'fog': pd.to_numeric(panel_raw.get('雾天数', pd.Series(0, index=panel_raw.index))).astype(float), 'severe': pd.to_numeric(panel_raw.get('严重事故数', pd.Series(0, index=panel_raw.index))).fillna(0).astype(int), 'injury': pd.to_numeric(panel_raw['伤人事故数']).fillna(0).astype(int), 'fatal': pd.to_numeric(panel_raw['死亡事故数']).fillna(0).astype(int), 'minor': pd.to_numeric(panel_raw['轻微事故数']).fillna(0).astype(int)})
    panel['date'] = pd.to_datetime(dict(year=panel['year'], month=panel['month'], day=1))
    if risk_values is not None:
        if len(risk_values) != len(base) or not np.isfinite(risk_values).all():
            raise ValueError('Provide a finite risk vector aligned with all base-sheet segments.')
        base['baseline_risk'] = np.asarray(risk_values, dtype=float)
    panel['maintenance'] = maintenance_for_panel(base['segment'], base['baseline_risk'], panel['segment'], panel['date'], **maintenance_parameters or {})
    panel['calendar'] = panel['date'].dt.strftime('%Y-%m')
    panel['month_of_year'] = panel['month'].astype(str)
    panel = panel.merge(base, on='segment', how='left', validate='many_to_one')
    if set(base['treated'].unique()) != {0, 1}:
        raise ValueError('Both treated and control segments are required.')
    if not np.isfinite(panel[['y', 'exposure', 'speed', 'baseline_risk']].to_numpy(float)).all():
        raise ValueError('Outcomes, exposure, speed and risk must be finite.')
    if panel['y'].lt(0).any() or panel['exposure'].le(0).any():
        raise ValueError('Crash counts must be nonnegative and exposure must be positive.')
    if base.loc[base['treated'].eq(1), 'effective_date'].isna().any():
        raise ValueError('Each treated segment requires an effective month.')
    panel['log_exposure'] = np.log(panel['exposure'])
    keys = panel[['segment', 'calendar']]
    expected_exposure = panel['aadt'] * panel['length_m'] / 1000 * panel['date'].dt.days_in_month
    audit = {'input_sha256': sha256(input_path), 'segments': int(base.shape[0]), 'panel_rows': int(panel.shape[0]), 'treated_segments': int(base['treated'].sum()), 'control_segments': int((base['treated'] == 0).sum()), 'duplicate_segment_months': int(keys.duplicated().sum()), 'missing_primary_outcomes': int(panel['y'].isna().sum()), 'total_crashes': int(panel['y'].sum()), 'zero_crash_months': int(panel['y'].eq(0).sum()), 'zero_crash_share': float(panel['y'].eq(0).mean()), 'maximum_monthly_crashes': int(panel['y'].max()), 'mean_monthly_crashes': float(panel['y'].mean()), 'variance_monthly_crashes': float(panel['y'].var(ddof=1)), 'variance_to_mean': float(panel['y'].var(ddof=1) / panel['y'].mean()), 'fatal_crashes': int(panel['fatal'].sum()), 'injury_crashes': int(panel['injury'].sum()), 'minor_crashes': int(panel['minor'].sum()), 'severity_partition_mismatches': int((panel['y'] != panel['fatal'] + panel['injury'] + panel['minor']).sum()), 'treated_effective_dates_present': int(base.loc[base['treated'].eq(1), 'effective_date'].notna().sum()), 'control_effective_dates_blank': int(base.loc[base['treated'].eq(0), 'effective_date'].isna().sum()), 'post_on_control_rows': int(panel.loc[panel['treated'].eq(0), 'D'].sum()), 'exposure_max_abs_error': float((panel['exposure'] - expected_exposure).abs().max()), 'treated_cost_yuan': float(base.loc[base['treated'].eq(1), 'cost_yuan'].sum()), 'rainfall_common_within_calendar_months': int(panel.groupby('calendar')['rainfall'].nunique().eq(1).sum()), 'fog_common_within_calendar_months': int(panel.groupby('calendar')['fog'].nunique().eq(1).sum())}
    return (base, panel, audit)

def construct_baseline_and_matching(base: pd.DataFrame, panel: pd.DataFrame):
    cutoff = base.loc[base['treated'].eq(1), 'effective_date'].min()
    common_pre = panel.loc[panel['date'].lt(cutoff)].copy()
    agg = common_pre.groupby('segment', as_index=False).agg(pre_crashes=('y', 'sum'), pre_exposure=('exposure', 'sum'), speed_baseline=('speed', 'mean'), maintenance_baseline=('maintenance', 'mean'), rainfall_baseline=('rainfall', 'mean'), fog_baseline=('fog', 'mean'))
    baseline = base.merge(agg, on='segment', how='left', validate='one_to_one')
    baseline['pre_rate_million_vkm'] = baseline['pre_crashes'] / baseline['pre_exposure'] * 1000000
    baseline['log_aadt'] = np.log(baseline['aadt'])
    baseline['log_length'] = np.log(baseline['length_m'])
    baseline['asinh_pre_rate'] = np.arcsinh(baseline['pre_rate_million_vkm'])
    match_covariates = ['asinh_pre_rate', 'log_aadt', 'log_length', 'lane_width', 'speed_baseline', 'maintenance_baseline', 'baseline_risk']
    X = baseline[match_covariates].astype(float)
    means, sds = (X.mean(), X.std(ddof=0).replace(0, 1))
    Xz = (X - means) / sds
    Xlogit = sm.add_constant(Xz)
    logit_result = sm.Logit(baseline['treated'], Xlogit).fit(disp=False, maxiter=500)
    baseline['propensity_score'] = logit_result.predict(Xlogit)
    p = np.clip(baseline['propensity_score'].to_numpy(), 1e-08, 1 - 1e-08)
    baseline['logit_propensity'] = np.log(p / (1 - p))
    treated = baseline.loc[baseline['treated'].eq(1)].copy()
    controls = baseline.loc[baseline['treated'].eq(0)].copy()
    pairs = []
    for _, tr in treated.iterrows():
        distances = (controls['logit_propensity'] - tr['logit_propensity']).abs()
        control_index = distances.idxmin()
        co = controls.loc[control_index]
        pairs.append({'treated_segment': tr['segment'], 'control_segment': co['segment'], 'treated_effective_date': tr['effective_date'], 'propensity_treated': tr['propensity_score'], 'propensity_control': co['propensity_score'], 'absolute_logit_distance': float(distances.loc[control_index])})
    pairs = pd.DataFrame(pairs)
    control_counts = pairs['control_segment'].value_counts()
    baseline['matched_weight'] = np.where(baseline['treated'].eq(1), 1.0, baseline['segment'].map(control_counts).fillna(0).astype(float))
    balance_vars = {'Pre-intervention crash rate (per million vehicle-km)': 'pre_rate_million_vkm', 'AADT': 'aadt', 'Segment length (m)': 'length_m', 'Lane width (m)': 'lane_width', '85th percentile speed': 'speed_baseline', 'Maintenance score': 'maintenance_baseline', 'Baseline risk index': 'baseline_risk'}
    t = baseline.loc[baseline['treated'].eq(1)]
    c = baseline.loc[baseline['treated'].eq(0)]
    balance_rows = []
    for label, col in balance_vars.items():
        before_smd = smd(t[col], c[col], pd.Series(1.0, index=t.index), pd.Series(1.0, index=c.index))
        after_smd = smd(t[col], c[col], t['matched_weight'], c['matched_weight'])
        t_before, _ = weighted_stats(t[col], pd.Series(1.0, index=t.index))
        c_before, _ = weighted_stats(c[col], pd.Series(1.0, index=c.index))
        t_after, _ = weighted_stats(t[col], t['matched_weight'])
        c_after, _ = weighted_stats(c[col], c['matched_weight'])
        balance_rows.append({'variable': label, 'treated_mean_before': t_before, 'control_mean_before': c_before, 'smd_before': before_smd, 'treated_mean_after': t_after, 'control_mean_after': c_after, 'smd_after': after_smd})
    balance = pd.DataFrame(balance_rows)
    matching_summary = {'common_pre_start': str(common_pre['date'].min().date()), 'common_pre_end': str(common_pre['date'].max().date()), 'matching_method': 'nearest_logit_with_replacement', 'treated_matched': int(len(pairs)), 'unique_controls_used': int(pairs['control_segment'].nunique()), 'maximum_control_reuse': int(control_counts.max()), 'maximum_absolute_logit_distance': float(pairs['absolute_logit_distance'].max()), 'median_absolute_logit_distance': float(pairs['absolute_logit_distance'].median()), 'propensity_model_converged': bool(logit_result.mle_retvals['converged']), 'propensity_model_coefficients': {k: float(v) for k, v in logit_result.params.items()}}
    return (baseline, pairs, balance, matching_summary)

def attach_baseline(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    cols = ['segment', 'matched_weight', 'pre_crashes', 'pre_rate_million_vkm', 'speed_baseline', 'maintenance_baseline', 'log_aadt', 'log_length']
    out = panel.merge(baseline[cols], on='segment', how='left', validate='many_to_one')
    for col in ['log_aadt', 'log_length', 'lane_width', 'speed_baseline', 'maintenance_baseline', 'baseline_risk']:
        mean = out.drop_duplicates('segment')[col].mean()
        sd = out.drop_duplicates('segment')[col].std(ddof=0)
        out[f'{col}_z'] = (out[col] - mean) / (sd if sd > 0 else 1)
    return out

def spf_empirical_bayes(panel: pd.DataFrame):
    controls = panel.loc[panel['treated'].eq(0)].copy()
    formula = 'y ~ log_aadt_z + log_length_z + lane_width_z + speed_baseline_z + maintenance_baseline_z + C(month_of_year)'
    model = fit_nb2(formula, controls)
    treated = panel.loc[panel['treated'].eq(1)].copy()
    treated['spf_mu'] = model.predict(treated, offset=treated['log_exposure'])
    segment_rows = []
    alpha = max(float(model.params.get('alpha', 0.0)), 0.0)
    for segment, d in treated.groupby('segment'):
        pre = d['D'].eq(0)
        p_pre = float(d.loc[pre, 'spf_mu'].sum())
        p_post = float(d.loc[~pre, 'spf_mu'].sum())
        o_pre = float(d.loc[pre, 'y'].sum())
        o_post = float(d.loc[~pre, 'y'].sum())
        weight = 1 / (1 + alpha * p_pre)
        eb_pre = weight * p_pre + (1 - weight) * o_pre
        eb_post = eb_pre * (p_post / p_pre) if p_pre > 0 else p_post
        segment_rows.append({'segment': segment, 'observed_pre': o_pre, 'observed_post': o_post, 'spf_predicted_pre': p_pre, 'spf_predicted_post': p_post, 'eb_weight': weight, 'eb_expected_post': eb_post})
    seg = pd.DataFrame(segment_rows)
    estimate = float(seg['observed_post'].sum() / seg['eb_expected_post'].sum())
    rng = np.random.default_rng(SEED + 1)
    idx = np.arange(len(seg))
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        s = seg.iloc[rng.choice(idx, size=len(idx), replace=True)]
        denom = s['eb_expected_post'].sum()
        if denom > 0:
            draws.append(float(s['observed_post'].sum() / denom))
    return ({'model': 'Control-SPF empirical Bayes before-after', 'measure': 'O/E ratio', 'estimate': estimate, 'ci_low': float(np.quantile(draws, 0.025)), 'ci_high': float(np.quantile(draws, 0.975)), 'dispersion_alpha': alpha, 'bootstrap_reps_valid': len(draws)}, seg, model)
