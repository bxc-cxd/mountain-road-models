from pathlib import Path
import os, sys, json, hashlib, importlib.util, time, warnings
for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[k] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from .config import INPUT, EXCLUDE_YEAR, folder
from . import core
WORK = folder('cache')
RESULTS = folder('models')
FORM = 'y ~ treated + D + log_aadt_z + log_length_z + lane_width_z + speed_baseline_z + maintenance_baseline_z + baseline_risk_z + C(road_class) + C(calendar)'
TREND = FORM.replace('treated + D +', 'treated + D + time_after +')
ALPHA_FORM = FORM.replace('C(calendar)', 'C(month_of_year)')
PS_COLS = ['asinh_pre_rate', 'log_aadt', 'log_length', 'lane_width', 'speed_baseline', 'maintenance_baseline', 'baseline_risk']
SEED = 20261105

def dump(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def prepare(exclude_year=None, *, risk_values=None, maintenance_parameters=None):
    b, p, audit = core.load_and_prepare(INPUT, risk_values=risk_values, maintenance_parameters=maintenance_parameters)
    if exclude_year is not None:
        p = p[p.year.ne(exclude_year)].copy()
    baseline, pairs, _, match = core.construct_baseline_and_matching(b, p)
    p = core.attach_baseline(p, baseline)
    for d in [baseline, p]:
        for col in d.select_dtypes(include=['str']).columns:
            d[col] = d[col].astype(object)
    e = baseline.propensity_score.to_numpy()
    T = baseline.treated.to_numpy()
    for col, v in [('att_weight', np.where(T == 1, 1, e / (1 - e))), ('overlap_weight', np.where(T == 1, 1 - e, e))]:
        baseline[col] = v / v.mean()
        p[col] = p.segment.map(baseline.set_index('segment')[col])
    return (b, baseline, p, pairs, audit, match)

def estimate_alpha(p):
    r = smf.negativebinomial(ALPHA_FORM, data=p, offset=p.log_exposure).fit(method='bfgs', maxiter=1000, disp=False)
    assert bool(r.mle_retvals['converged']), r.mle_retvals
    return (max(float(r.params['alpha']), 1e-08), r)

def fit_gee(p, alpha, weight='att_weight', trend=True, poisson=False):
    m = smf.gee(TREND if trend else FORM, groups='segment', data=p, family=sm.families.Poisson() if poisson else sm.families.NegativeBinomial(alpha=alpha), cov_struct=sm.cov_struct.Independence(), offset=p.log_exposure, weights=p[weight])
    r = m.fit(maxiter=500, ctol=1e-10)
    assert r.converged and np.isfinite(r.params).all()
    return r

def contrast(r, p, name, weight='att_weight', trend=True, term=None):
    c = np.zeros(len(r.params))
    names = list(r.params.index)
    if term is not None:
        c[names.index(term)] = 1
        h = None
    else:
        c[names.index('D')] = 1
        h = float(np.average(p.loc[p.D.eq(1), 'time_after'], weights=p.loc[p.D.eq(1), weight])) if trend else None
        if trend:
            c[names.index('time_after')] = h
    b = float(c @ r.params)
    se = float(np.sqrt(c @ r.cov_params() @ c))
    return {'analysis': name, 'term': term or ('average observed post-period effect' if trend else 'D'), 'coefficient': b, 'std_error': se, 'irr': float(np.exp(b)), 'ci_low': float(np.exp(b - norm.ppf(0.975) * se)), 'ci_high': float(np.exp(b + norm.ppf(0.975) * se)), 'p_value': float(2 * norm.sf(abs(b / se))), 'n_obs': len(p), 'segments': int(p.segment.nunique()), 'treated_segments': int(p.loc[p.treated.eq(1), 'segment'].nunique()), 'post_month_mean': h, 'converged': bool(r.converged)}

def weighted_nb_alpha(p, r, weight='att_weight'):
    X = np.asarray(r.model.exog)
    y = np.asarray(r.model.endog)
    off = np.asarray(p.log_exposure)
    w = np.asarray(p[weight])
    scale = w.sum()

    def fg(z):
        eta = off + X @ z[:-1]
        mu = np.exp(eta)
        a = np.exp(z[-1])
        rr = 1 / a
        ll = gammaln(y + rr) - gammaln(rr) - gammaln(y + 1) - rr * np.log1p(a * mu) + y * (np.log(a) + eta - np.log1p(a * mu))
        gb = X.T @ (w * (y - mu) / (1 + a * mu))
        dr = digamma(y + rr) - digamma(rr) - np.log1p(a * mu) + 1 - (rr + y) / (rr + mu)
        ga = np.sum(w * (-rr * dr))
        return (-float(w @ ll) / scale, -np.r_[gb, ga] / scale)
    opt = minimize(fg, np.r_[r.params, np.log(max(float(r.model.family.alpha), 1e-05))], method='L-BFGS-B', jac=True, bounds=[(None, None)] * X.shape[1] + [(np.log(1e-05), np.log(100))], options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-09, 'maxls': 50})
    return (float(np.exp(opt.x[-1])), {'success': bool(opt.success), 'message': str(opt.message), 'nit': int(opt.nit), 'max_abs_gradient': float(np.max(np.abs(opt.jac))), 'alpha': float(np.exp(opt.x[-1])), 'at_lower_boundary': bool(opt.x[-1] < np.log(1e-05) + 1e-05)})

def balance(b):
    t = b[b.treated.eq(1)]
    c = b[b.treated.eq(0)]
    rows = []
    cols = ['pre_rate_million_vkm', 'aadt', 'length_m', 'lane_width', 'speed_baseline', 'maintenance_baseline', 'baseline_risk']
    for v in cols:
        row = {'variable': v, 'treated_unweighted': float(t[v].mean()), 'control_unweighted': float(c[v].mean())}
        for wt in ['att_weight', 'overlap_weight']:
            row[wt + '_treated'] = float(np.average(t[v], weights=t[wt]))
            row[wt + '_control'] = float(np.average(c[v], weights=c[wt]))
            row[wt + '_smd'] = core.smd(t[v], c[v], t[wt], c[wt])
        rows.append(row)
    w = c.att_weight.to_numpy()
    return (rows, {'control_ess': float(w.sum() ** 2 / (w * w).sum()), 'max_control_weight_normalized': float(w.max()), 'max_abs_att_smd': float(max((abs(r['att_weight_smd']) for r in rows))), 'max_abs_overlap_smd': float(max((abs(r['overlap_weight_smd']) for r in rows))), 'ps_min': float(b.propensity_score.min()), 'ps_max': float(b.propensity_score.max())})

def before_after(p, pairs):
    pieces = []
    by = {k: d for k, d in p.groupby('segment')}
    for rr in pairs.itertuples():
        ar = []
        for seg in [rr.treated_segment, rr.control_segment]:
            d = by[seg]
            pre = d.date.lt(rr.treated_effective_date)
            ar.extend([d.loc[pre, 'y'].sum(), d.loc[pre, 'exposure'].sum(), d.loc[~pre, 'y'].sum(), d.loc[~pre, 'exposure'].sum()])
        pieces.append(ar)
    A = np.asarray(pieces, dtype=float)

    def calc(x):
        return x[..., 2] / x[..., 3] / (x[..., 0] / x[..., 1]) / (x[..., 6] / x[..., 7] / (x[..., 4] / x[..., 5]))
    rng = np.random.default_rng(core.SEED)
    idx = rng.choice(len(A), size=(core.BOOTSTRAP_REPS, len(A)), replace=True)
    draws = calc(A[idx].sum(axis=1))
    point = calc(A.sum(axis=0))
    tot = A.sum(axis=0)
    return {'analysis': 'Matched comparison-group before-after', 'irr': float(point), 'ci_low': float(np.quantile(draws, 0.025)), 'ci_high': float(np.quantile(draws, 0.975)), 'bootstrap_reps': len(draws), 'pairs': len(A), 'unique_controls': int(pairs.control_segment.nunique()), 'treated_pre_crashes': tot[0], 'treated_pre_exposure': tot[1], 'treated_post_crashes': tot[2], 'treated_post_exposure': tot[3], 'control_pre_crashes_counting_reuse': tot[4], 'control_pre_exposure_counting_reuse': tot[5], 'control_post_crashes_counting_reuse': tot[6], 'control_post_exposure_counting_reuse': tot[7], 'treated_rate_ratio': float(tot[2] / tot[3] / (tot[0] / tot[1])), 'control_rate_ratio': float(tot[6] / tot[7] / (tot[4] / tot[5]))}

def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    b, baseline, p, pairs, audit, match = prepare()
    dump('data audit.json', audit)
    dump('matching.json', match)
    alpha, ar = estimate_alpha(p)
    results = []
    fits = {}
    for wt in ['att_weight', 'overlap_weight']:
        for trend in [True, False]:
            r = fit_gee(p, alpha, wt, trend)
            name = f'{wt}_NB_' + ('trend' if trend else 'level')
            fits[name] = r
            results.append(contrast(r, p, name, wt, trend))
            if trend:
                results.extend([contrast(r, p, name, wt, trend, term=z) for z in ['D', 'time_after']])
        r = fit_gee(p, alpha, wt, True, True)
        name = f'{wt}_Poisson_trend'
        fits[name] = r
        results.append(contrast(r, p, name, wt, True))
    primary = fits['att_weight_NB_trend']
    primary.params.to_csv(RESULTS / 'primary parameters.csv')
    primary.cov_params().to_csv(RESULTS / 'primary covariance.csv')
    br, bm = balance(baseline)
    pd.DataFrame(br).to_csv(RESULTS / 'balance.csv', index=False)
    dump('balance metadata.json', bm)
    baseline.to_pickle(WORK / 'baseline.pkl')
    p.to_pickle(WORK / 'panel.pkl')
    pairs.to_pickle(WORK / 'pairs.pkl')
    np.savez_compressed(WORK / 'bootstrap design.npz', X=np.asarray(primary.model.exog), Xalpha=np.asarray(ar.model.exog), beta=np.asarray(primary.params), alpha_beta=np.asarray(ar.params[:-1]), alpha=alpha, y=p.y.to_numpy(), offset=p.log_exposure.to_numpy(), sid=pd.Categorical(p.segment, categories=baseline.segment).codes, ps_X=np.c_[np.ones(len(baseline)), ((baseline[PS_COLS] - baseline[PS_COLS].mean()) / baseline[PS_COLS].std(ddof=0)).to_numpy()], treated=baseline.treated.to_numpy(), D=p.D.to_numpy(), time_after=p.time_after.to_numpy(), D_index=list(primary.params.index).index('D'), trend_index=list(primary.params.index).index('time_after'))
    limited = p[p.D.eq(0) | p.time_after.between(0, 11)].copy()
    positive = limited.groupby('segment').y.sum()
    common = limited[limited.segment.isin(positive[positive.gt(0)].index)].copy()
    rr = fit_gee(common, alpha, trend=False)
    results.append(contrast(rr, common, 'Common sample first 12 months ATT NB-GEE', trend=False))
    fm = smf.glm('y ~ D + C(segment) + C(calendar)', data=common, family=sm.families.Poisson(), offset=common.log_exposure, freq_weights=common.att_weight)
    fr = fm.fit(maxiter=500, cov_type='cluster', cov_kwds={'groups': common.segment})
    assert fr.converged
    results.append(contrast(fr, common, 'Common sample first 12 months ATT FE-PPML', trend=False))
    dump('common comparison.json', {'segments': common.segment.nunique(), 'treated_segments': common.loc[common.treated.eq(1), 'segment'].nunique(), 'observations': len(common), 'treated_post_observations': int(common.D.sum()), 'zero_crash_segments_excluded': len(b) - len(positive[positive.gt(0)]), 'weights': 'att_weight'})
    exclude_year = EXCLUDE_YEAR or int(b.loc[b.treated.eq(1), 'effective_date'].dt.year.min())
    _, bl2, p2, _, au2, ma2 = prepare(exclude_year)
    a2, _ = estimate_alpha(p2)
    r2 = fit_gee(p2, a2)
    results.append(contrast(r2, p2, f'ATT NB-GEE excluding {exclude_year}'))
    dump('excluded year.json', {'excluded_year': exclude_year, 'alpha': a2, 'baseline_start': ma2['common_pre_start'], 'baseline_end': ma2['common_pre_end'], 'audit': au2, 'balance': balance(bl2)[1]})
    aw, mw = weighted_nb_alpha(p, primary)
    dump('weighted dispersion.json', mw)
    assert mw['success'] and mw['max_abs_gradient'] < 1e-05, mw
    rw = fit_gee(p, aw)
    results.append(contrast(rw, p, 'ATT NB-GEE weighted mean-matched dispersion'))
    ba = before_after(p, pairs)
    dump('before after.json', ba)
    eb, seg, spf = core.spf_empirical_bayes(p)
    dump('spf eb.json', eb)
    seg.to_csv(RESULTS / 'spf eb segments.csv', index=False)
    spf.params.to_csv(RESULTS / 'spf parameters.csv')
    dump('spf metadata.json', {'formula': spf.model.formula, 'alpha': float(spf.params['alpha']), 'n_obs': int(spf.nobs), 'n_segments': int(b.treated.eq(0).sum()), 'calibration_start': str(p.loc[p.treated.eq(0), 'date'].min().date()), 'calibration_end': str(p.loc[p.treated.eq(0), 'date'].max().date()), 'pre_expected_spf': seg.spf_predicted_pre.sum(), 'post_expected_spf': seg.spf_predicted_post.sum(), 'pre_observed': seg.observed_pre.sum(), 'post_observed': seg.observed_post.sum(), 'eb_post_expected': seg.eb_expected_post.sum(), 'converged': bool(spf.mle_retvals['converged'])})
    pd.DataFrame(results).to_csv(RESULTS / 'model results.csv', index=False)
    dump('run metadata.json', {'input': str(INPUT), 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(), 'alpha': alpha, 'elapsed_seconds': time.time() - t0, 'primary': 'ATT weighted NB GEE with trend, geometric mean over observed post segment-months', 'seed': SEED, 'approximate_primary_MDE_reduction': float(1 - np.exp(-(norm.ppf(0.975) + norm.ppf(0.8)) * results[0]['std_error']))})
if __name__ == '__main__':
    main()
