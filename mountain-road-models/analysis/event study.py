from pathlib import Path
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import hashlib
import importlib.util
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, chi2
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .config import INPUT as SOURCE, folder
from . import core
RESULTS = folder('event')
WORK = folder('cache')
BINS = ['pre_early', 'pre_12_7', 'pre_6_2', 'pre_1', 'post_0_2', 'post_3_6', 'post_7_12', 'post_13']
LABELS = ['≤−13', '−12 to −7', '−6 to −2', '−1', '0 to 2', '3 to 6', '7 to 12', '≥13']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_json(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def mark_bins(d):
    d = d.copy()
    e = d.event_time
    t = d.treated.eq(1)
    conditions = [e.le(-13), e.between(-12, -7), e.between(-6, -2), e.eq(-1), e.between(0, 2), e.between(3, 6), e.between(7, 12), e.ge(13)]
    for b, c in zip(BINS, conditions):
        d[b] = (t & c).astype(float)
    assert (d[BINS].sum(axis=1) == d.treated).all()
    return d

def fitted_panel(d):
    totals = d.groupby('segment').y.sum()
    return d[d.segment.isin(totals[totals.gt(0)].index)].copy()

def fit(d, terms):
    d = d.copy()
    d['segment'] = d['segment'].astype(object)
    d['calendar'] = d['calendar'].astype(object)
    formula = 'y ~ ' + ' + '.join(terms) + ' + C(segment) + C(calendar)'
    model = smf.glm(formula, data=d, family=sm.families.Poisson(), offset=d.log_exposure)
    rank = np.linalg.matrix_rank(model.exog)
    assert rank == model.exog.shape[1], (rank, model.exog.shape)
    result = model.fit(maxiter=500, method='irls', cov_type='cluster', cov_kwds={'groups': d.segment})
    assert result.converged and np.isfinite(result.params).all()
    return result

def event_fit(panel, name):
    data = mark_bins(fitted_panel(panel))
    present = [b for b in BINS if data[b].sum() > 0]
    ref_counts = data[['pre_12_7', 'pre_6_2']].sum()
    ref_w = ref_counts / ref_counts.sum()
    pivot = 'pre_6_2'
    terms = []
    for b in present:
        if b == pivot:
            continue
        col = 'x_' + b
        data[col] = data[b] - float(ref_w.get(b, 0) / ref_w[pivot]) * data[pivot]
        terms.append(col)
    result = fit(data, terms)
    names = list(result.params.index)
    n = len(names)
    C = []
    rows = []
    for b, label in zip(BINS, LABELS):
        all_b = mark_bins(panel)
        full = all_b[all_b[b].eq(1)]
        retained = data[data[b].eq(1)]
        row = {'analysis': name, 'window_id': b, 'window': label, 'crashes': int(full.y.sum()), 'segment_months': len(full), 'segments': int(full.segment.nunique()), 'estimation_segment_months': len(retained), 'estimation_segments': int(retained.segment.nunique()), 'exposure': float(full.exposure.sum())}
        if b not in present:
            row.update(coefficient=None, std_error=None, ci_low=None, ci_high=None, irr=None, irr_low=None, irr_high=None, p_value=None)
            rows.append(row)
            continue
        c = np.zeros(n)
        if b == pivot:
            for rb in ref_w.index:
                if rb != pivot:
                    c[names.index('x_' + rb)] = -float(ref_w[rb] / ref_w[pivot])
        else:
            c[names.index('x_' + b)] = 1.0
        beta = float(c @ result.params)
        v = float(c @ result.cov_params() @ c)
        assert v > 0
        se = np.sqrt(v)
        lo = beta - 1.959963984540054 * se
        hi = beta + 1.959963984540054 * se
        row.update(coefficient=beta, std_error=float(se), ci_low=lo, ci_high=hi, irr=float(np.exp(beta)), irr_low=float(np.exp(lo)), irr_high=float(np.exp(hi)), p_value=float(2 * norm.sf(abs(beta / se))))
        rows.append(row)
        if b.startswith('pre_') and b != pivot:
            C.append(c)
    C = np.asarray(C)
    test = result.wald_test(C, scalar=True, use_f=False)
    metadata = {'analysis': name, 'n_obs': int(result.nobs), 'n_segments': int(data.segment.nunique()), 'treated_segments': int(data.loc[data.treated.eq(1), 'segment'].nunique()), 'retained_crashes': int(data.y.sum()), 'excluded_zero_crash_segments': int(panel.segment.nunique() - data.segment.nunique()), 'reference': 'segment-month-weighted mean of coefficients for months -12 to -2', 'reference_weights': ref_w.to_dict(), 'pre_joint_chi2': float(test.statistic), 'pre_joint_df': int(np.linalg.matrix_rank(C)), 'pre_joint_p': float(test.pvalue), 'converged': bool(result.converged), 'score_max_abs': float(np.max(np.abs(result.model.score(result.params)))), 'formula': result.model.formula}
    result.params.to_csv(RESULTS / f'{name} parameters.csv'.replace('_', ' '))
    result.cov_params().to_csv(RESULTS / f'{name} covariance.csv'.replace('_', ' '))
    omitted = 'pre_1' if 'pre_1' in present else pivot
    alternate_terms = [b for b in present if b != omitted]
    alternate = fit(data, alternate_terms)
    aw = np.zeros(len(alternate.params))
    an = list(alternate.params.index)
    for rb, rw in ref_w.items():
        if rb != omitted:
            aw[an.index(rb)] = rw
    errs = []
    se_errs = []
    for row in rows:
        b = row['window_id']
        if b not in present:
            continue
        c = -aw.copy()
        if b != omitted:
            c[an.index(b)] += 1
        bb = float(c @ alternate.params)
        ss = float(np.sqrt(c @ alternate.cov_params() @ c))
        errs.append(abs(bb - row['coefficient']))
        se_errs.append(abs(ss - row['std_error']))
    assert max(errs) < 1e-07 and max(se_errs) < 1e-07
    metadata['independent_normalization_max_beta_difference'] = max(errs)
    metadata['independent_normalization_max_se_difference'] = max(se_errs)
    if name == 'all_months':
        alternate_pre = [b for b in alternate_terms if b.startswith('pre_')]
        alternate_C = np.zeros((len(alternate_pre), len(an)))
        for i, b in enumerate(alternate_pre):
            alternate_C[i, an.index(b)] = 1
        alternate_test = alternate.wald_test(alternate_C, scalar=True, use_f=False)
        metadata['alternate_reference_pre_joint_chi2'] = float(alternate_test.statistic)
        metadata['alternate_reference_pre_joint_p'] = float(alternate_test.pvalue)
        pd.DataFrame([{'window': b, 'beta': float(alternate.params[b]), 'se': float(alternate.bse[b])} for b in alternate_terms]).to_csv(RESULTS / 'alternate reference.csv', index=False)
    return (pd.DataFrame(rows), metadata)

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    base, panel, audit = core.load_and_prepare(SOURCE)
    raw = pd.read_excel(SOURCE, sheet_name='路段基础数据')
    dates = pd.DataFrame({'segment': raw['路段编号'].astype(str), 'work_start': pd.to_datetime(raw['干预开始日期'], errors='coerce'), 'work_end': pd.to_datetime(raw['干预结束日期'], errors='coerce')})
    base = base.merge(dates, on='segment', validate='one_to_one')
    panel = panel.merge(dates, on='segment', validate='many_to_one')
    tb = base[base.treated.eq(1)].copy()
    assert tb[['effective_date', 'work_start', 'work_end']].notna().all().all()
    assert (tb.work_start <= tb.work_end).all()
    assert (tb.effective_date.dt.to_period('M') == tb.work_end.dt.to_period('M') + 1).all()
    expected = (panel.date.dt.year - panel.effective_date.dt.year) * 12 + panel.date.dt.month - panel.effective_date.dt.month
    tr = panel.treated.eq(1)
    assert np.array_equal(panel.loc[tr, 'event_time'].values, expected[tr].values)
    assert (panel.D == (tr & expected.ge(0)).astype(int)).all()
    construction = tr & panel.date.between(panel.work_start, panel.work_end)
    post_count = panel.loc[tr & panel.D.eq(1), 'y'].sum()
    selected = panel[construction]
    clean = panel[~construction].copy()
    audit.update(construction_months=int(construction.sum()), construction_crashes=int(selected.y.sum()), construction_segments=int(selected.segment.nunique()), construction_duration_months=((tb.work_end.dt.year - tb.work_start.dt.year) * 12 + tb.work_end.dt.month - tb.work_start.dt.month + 1).value_counts().sort_index().to_dict(), start_min=str(tb.work_start.min().date()), start_max=str(tb.work_start.max().date()), end_min=str(tb.work_end.min().date()), end_max=str(tb.work_end.max().date()), effective_month_next_after_end=True, environment={'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__})
    save_json(RESULTS / 'input audit.json', audit)
    tb[['segment', 'work_start', 'work_end', 'effective_date']].to_csv(RESULTS / 'construction dates.csv', index=False)
    selected[['segment', 'date', 'event_time', 'y', 'exposure']].to_csv(RESULTS / 'construction months.csv', index=False)
    frames = []
    meta = []
    for d, name in [(panel, 'all_months'), (clean, 'excluding_construction')]:
        f, m = event_fit(d, name)
        frames.append(f)
        meta.append(m)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(RESULTS / 'event results.csv', index=False, encoding='utf-8-sig')
    save_json(RESULTS / 'event metadata.json', meta)
    months = panel.loc[tr].groupby('date').D.mean()
    controls = panel.loc[~tr].copy()
    controls['post_share'] = controls.date.map(months)
    controls['pre_share'] = 1 - controls.post_share
    table = []
    for group, period in [('Control', 'Pre-intervention'), ('Control', 'Post-intervention'), ('Treated', 'Pre-intervention'), ('Treated', 'Post-intervention')]:
        post = period.startswith('Post')
        if group == 'Control':
            d = controls
            w = d.post_share if post else d.pre_share
        else:
            d = panel[tr & panel.D.eq(int(post))]
            w = np.ones(len(d))
        n = float(np.sum(w))
        cr = float(np.sum(d.y * w))
        ex = float(np.sum(d.exposure * w))
        table.append({'Group': group, 'Period': period, 'Observations': n, 'Segments': int(d.segment.nunique()), 'Total crashes': cr, 'Mean monthly crashes': cr / n, 'Crashes per million vehicle-km': cr / ex * 1000000.0, 'exposure': ex})
    assert abs(sum((x['Observations'] for x in table[:2])) - len(controls)) < 1e-08
    assert abs(sum((x['Total crashes'] for x in table[:2])) - controls.y.sum()) < 1e-08
    for pi, post in enumerate([False, True]):
        n = cr = ex = 0.0
        for g, cnt in tb.groupby('effective_date').size().items():
            d = controls[controls.date.ge(g) if post else controls.date.lt(g)]
            w = cnt / len(tb)
            n += w * len(d)
            cr += w * d.y.sum()
            ex += w * d.exposure.sum()
        assert np.allclose([n, cr, ex], [table[pi]['Observations'], table[pi]['Total crashes'], table[pi]['exposure']], rtol=1e-12, atol=1e-08)
    pd.DataFrame(table).to_csv(RESULTS / 'calendar aligned summary.csv', index=False, encoding='utf-8-sig')
    controls.groupby('date').agg(pre_weight=('pre_share', 'first'), post_weight=('post_share', 'first'), crashes=('y', 'sum'), exposure=('exposure', 'sum')).to_csv(RESULTS / 'control calendar weights.csv')
    risk = base.loc[base.treated.eq(1), ['segment', 'baseline_risk']].copy()
    risk['risk_stratum'] = pd.qcut(risk.baseline_risk.rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    traj = panel[tr & panel.event_time.between(-24, 24)].merge(risk[['segment', 'risk_stratum']], on='segment', validate='many_to_one')
    traj = traj.groupby(['risk_stratum', 'event_time'], observed=True).agg(crashes=('y', 'sum'), exposure=('exposure', 'sum'), segment_months=('y', 'size')).reset_index()
    traj['rate'] = traj.crashes / traj.exposure * 1000000.0
    traj.to_csv(RESULTS / 'risk stratum rates.csv', index=False, encoding='utf-8-sig')
    assert sha(SOURCE) == source_hash
if __name__ == '__main__':
    main()
