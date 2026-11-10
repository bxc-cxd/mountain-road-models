from pathlib import Path
import os, sys, json, hashlib, importlib.util, time
for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[key] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats
from .config import INPUT as SOURCE, REFERENCE_YEAR, folder
from . import models
WORK = folder('economics')
OUT = WORK
MODEL_RESULTS = folder('models')
SEED, DRAWS = (20260751, 5000)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(name, obj):

    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (Path, pd.Timestamp)):
            return str(x)
        raise TypeError(type(x))
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding='utf-8')

def save_csv(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def interval(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]

def ranked(frame, kind='three'):
    d = frame.copy()
    risk = d.baseline_risk.rank(method='average')
    benefit = d.benefit.rank(method='average')
    ce = d.ce.rank(method='average')
    if kind == 'three':
        d['_score_rank_sum'] = risk + benefit + ce
    elif kind == 'risk_ce':
        d['_score_rank_sum'] = risk + ce
    elif kind == 'ce':
        d['_score_rank_sum'] = ce
    else:
        raise ValueError(kind)
    d['score'] = d['_score_rank_sum'] / len(d)
    d = d.sort_values(['_score_rank_sum', 'benefit', 'segment'], ascending=[False, False, True]).drop(columns='_score_rank_sum').reset_index(drop=True)
    d['rank'] = np.arange(1, len(d) + 1)
    return d

def compare_ranks(ref, other, label):
    a = ref.set_index('segment')['rank']
    b = other.set_index('segment')['rank'].reindex(a.index)
    return {'comparison': label, 'spearman_r': float(stats.spearmanr(a, b).statistic), 'top10_overlap': len(set(ref.segment.head(10)) & set(other.segment.head(10))), 'median_absolute_rank_change': float(np.median(np.abs(a - b))), 'maximum_absolute_rank_change': int(np.max(np.abs(a - b))), 'entrants_to_top10': sorted(set(other.segment.head(10)) - set(ref.segment.head(10))), 'exits_from_top10': sorted(set(ref.segment.head(10)) - set(other.segment.head(10)))}

def benefit_data(result, rows, ids, draws):
    d = rows.copy()
    counter = d.copy()
    counter['D'] = 0
    counter['time_after'] = 0
    d['mu0'] = result.predict(counter, offset=counter.log_exposure)
    d['mu1'] = result.predict(d, offset=d.log_exposure)
    h = d.time_after.astype(int)
    assert np.allclose(h, d.time_after)
    matrix = d.assign(k=h).pivot(index='segment', columns='k', values='mu0').reindex(index=ids).fillna(0)
    k = matrix.columns.to_numpy(dtype=float)
    effects = result.params['D'] + result.params['time_after'] * k
    benefit = matrix.to_numpy() @ (1 - np.exp(effects))
    simulated = (1 - np.exp(draws[:, [0]] + draws[:, [1]] * k[None, :])) @ matrix.to_numpy().T
    direct = (d.mu0 - d.mu1).groupby(d.segment).sum().reindex(ids).to_numpy()
    assert np.allclose(benefit, direct, rtol=1e-11, atol=1e-11)
    return (benefit, simulated, d, matrix)

def economic_summary(frame, simulated, label, nmonths):
    total = float(frame.benefit.sum())
    cost = float(frame.cost_yuan.sum())
    low, high = interval(simulated.sum(axis=1))
    return {'horizon': label, 'segments': len(frame), 'segment_months': int(nmonths), 'engineering_cost_yuan': cost, 'benefit': total, 'benefit_ci_low': low, 'benefit_ci_high': high, 'benefit_per_segment': total / len(frame), 'crashes_per_10000_yuan': total / cost * 10000, 'ce_ci_low': low / cost * 10000, 'ce_ci_high': high / cost * 10000, 'cost_per_crash_yuan': cost / total if total > 0 else None, 'interval_method': 'conditional_joint_normal', 'simulation_draws': DRAWS, 'seed': SEED}

def speed_analysis(raw_b, raw_m):
    ids = raw_b.loc[raw_b['处理组标记'].eq(1) & raw_b['速度管理'].eq(1), '路段编号']
    data = raw_m.copy()
    data['date'] = pd.to_datetime(data['日历月份'])
    base = raw_b.set_index('路段编号')
    rows = []
    for sid in sorted(ids):
        start = pd.Timestamp(base.at[sid, '干预开始日期']).to_period('M').to_timestamp()
        effective = pd.Timestamp(base.at[sid, '干预生效月份']).to_period('M').to_timestamp()
        d = data[data['路段编号'].eq(sid)]
        pre = d[d.date.ge(start - pd.DateOffset(months=12)) & d.date.lt(start)]
        post = d[d.date.ge(effective) & d.date.lt(effective + pd.DateOffset(months=12))]
        assert len(pre) == len(post) == 12 and pre['第85百分位速度'].notna().all() and post['第85百分位速度'].notna().all()
        before = float(pre['第85百分位速度'].mean())
        after = float(post['第85百分位速度'].mean())
        rows.append({'segment': sid, 'construction_start': start, 'effective_month': effective, 'pre_n': len(pre), 'post_n': len(post), 'pre_mean_kmh': before, 'post_mean_kmh': after, 'difference_kmh': after - before, 'installed_measures': base.at[sid, '原始治理措施']})
    a = pd.DataFrame(rows)
    x = a.difference_kmh.to_numpy()
    n = len(x)
    sw = stats.shapiro(x)
    normality = {'test': 'Shapiro-Wilk', 'statistic': float(sw.statistic), 'p_value': float(sw.pvalue), 'n': len(x)}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    test = stats.ttest_1samp(x, 0)
    rng = np.random.default_rng(20261105)
    boot = x[rng.integers(n, size=(10000, n))].mean(axis=1)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    out = {'segments': n, 'pre_segment_months': int(a.pre_n.sum()), 'post_segment_months': int(a.post_n.sum()), 'pre_mean_kmh': float(a.pre_mean_kmh.mean()), 'post_mean_kmh': float(a.post_mean_kmh.mean()), 'pre_sd_across_segment_means': float(a.pre_mean_kmh.std(ddof=1)), 'post_sd_across_segment_means': float(a.post_mean_kmh.std(ddof=1)), 'difference_kmh': mean, 'difference_sd': sd, 'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'relative_change_percent': mean / a.pre_mean_kmh.mean() * 100, 't_statistic': float(test.statistic), 'df': n - 1, 'p_value': float(test.pvalue), 'cohen_dz': mean / sd, 'bootstrap_ci': interval(boot), 'bootstrap_reps': 10000, 'negative_differences': int((x < 0).sum()), 'positive_differences': int((x > 0).sum()), 'difference_median': float(np.median(x)), 'difference_min': float(x.min()), 'difference_max': float(x.max()), 'normality': normality, 'iqr_outliers': int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())}
    save_csv('speed segment changes.csv', a)
    dump('speed summary.json', out)
    return out

def main():
    from .config import require_inputs
    require_inputs('economics')
    started = time.time()
    hashes = {'input': sha(SOURCE)}
    meta = json.loads((MODEL_RESULTS / 'run metadata.json').read_text(encoding='utf-8'))
    assert hashes['input'] == meta['input_sha256']
    raw_b = pd.read_excel(SOURCE, sheet_name='路段基础数据')
    raw_m = pd.read_excel(SOURCE, sheet_name='月度面板数据')
    b, baseline, p, pairs, audit, _ = models.prepare()
    r = models.fit_gee(p, meta['alpha'])
    saved_beta = pd.read_csv(MODEL_RESULTS / 'primary parameters.csv', index_col=0).iloc[:, 0]
    saved_cov = pd.read_csv(MODEL_RESULTS / 'primary covariance.csv', index_col=0)
    assert np.max(np.abs(r.params - saved_beta.reindex(r.params.index))) < 1e-08
    assert np.max(np.abs(r.cov_params().to_numpy() - saved_cov.loc[r.params.index, r.params.index].to_numpy())) < 1e-07
    point = models.contrast(r, p, 'ATT NB-GEE')
    terms = ['D', 'time_after']
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(r.params.loc[terms], r.cov_params().loc[terms, terms], size=DRAWS)
    ids = sorted(b.loc[b.treated.eq(1), 'segment'])
    static = b.set_index('segment').loc[ids, ['cost_yuan', 'baseline_risk']].reset_index()
    assert static.cost_yuan.gt(0).all()
    observed = p[p.D.eq(1)].copy()
    first = observed[observed.time_after.between(0, 11)].copy()
    assert first.groupby('segment').size().eq(12).all() and len(first) == 12 * len(ids)
    ben_all, sim_all, month_all, _ = benefit_data(r, observed, ids, draws)
    ben12, sim12, month12, _ = benefit_data(r, first, ids, draws)
    allseg = static.assign(benefit=ben_all)
    allseg['ce'] = allseg.benefit / allseg.cost_yuan * 10000
    seg = static.assign(benefit=ben12)
    seg['ce'] = seg.benefit / seg.cost_yuan * 10000
    seg['benefit_ci_low'] = np.quantile(sim12, 0.025, axis=0)
    seg['benefit_ci_high'] = np.quantile(sim12, 0.975, axis=0)
    seg['ce_ci_low'] = seg.benefit_ci_low / seg.cost_yuan * 10000
    seg['ce_ci_high'] = seg.benefit_ci_high / seg.cost_yuan * 10000
    seg['followup_months'] = seg.segment.map(observed.groupby('segment').size())
    raw_by = raw_b.set_index('路段编号')
    seg['installed_measures'] = seg.segment.map(raw_by['原始治理措施']).fillna('Not recorded')
    seg['cost_per_crash_yuan'] = seg.cost_yuan / seg.benefit
    summary_all = economic_summary(allseg, sim_all, 'Observed follow-up', len(observed))
    summary12 = economic_summary(seg, sim12, 'First 12 effective months (0-11)', len(first))
    ranks = {k: ranked(seg, k) for k in ['three', 'risk_ce', 'ce']}
    rank_all = ranked(allseg)
    comparisons = [compare_ranks(rank_all, ranks['three'], 'Observed follow-up to common 12 months (three indicators)')]
    comparisons += [compare_ranks(ranks['three'], ranks[k], f'Common 12 months: three indicators versus {k}') for k in ['risk_ce', 'ce']]
    freq = {k: np.zeros(len(ids)) for k in ranks}
    riskrank = stats.rankdata(seg.baseline_risk)
    cost = seg.cost_yuan.to_numpy()
    for vals in sim12:
        br = stats.rankdata(vals)
        cr = stats.rankdata(vals / cost)
        for k, scores in [('three', riskrank + br + cr), ('risk_ce', riskrank + cr), ('ce', cr)]:
            order = np.lexsort((np.asarray(ids), -vals, -scores))
            freq[k][order[:10]] += 1
    for k, d in ranks.items():
        d['top10_frequency'] = d.segment.map(dict(zip(ids, freq[k] / DRAWS)))
        d['rank_observed_followup'] = d.segment.map(rank_all.set_index('segment')['rank'])
        save_csv(f'treated ranking 12months {k}.csv'.replace('_', ' '), d)
    save_csv('treated ranking observed followup.csv', rank_all)
    save_csv('economic summary.csv', pd.DataFrame([summary_all, summary12]))
    dump('ranking comparisons.json', comparisons)
    save_csv('ranking comparisons.csv', pd.DataFrame(comparisons))
    save_csv('treated monthly predictions 12months.csv', month12[['segment', 'date', 'time_after', 'y', 'exposure', 'mu0', 'mu1']])
    estimates = pd.read_csv(MODEL_RESULTS / 'model results.csv')
    comparisons_fit = pd.read_csv(MODEL_RESULTS / 'comparisons.csv')
    before_after = json.loads((MODEL_RESULTS / 'before after.json').read_text(encoding='utf-8'))
    fe = comparisons_fit.loc[comparisons_fit.analysis.eq('Segment and calendar-month FE PPML') & comparisons_fit.term.eq('D'), 'estimate']
    overlap = estimates.loc[estimates.analysis.eq('overlap_weight_NB_trend') & estimates.term.eq('average observed post-period effect'), 'irr']
    if len(fe) != 1 or len(overlap) != 1:
        raise ValueError('Expected one fixed-effects and one overlap estimate.')
    scenarios = []
    values = [('ATT observed-period average', point['irr']), ('Full segment FE PPML', float(fe.iloc[0])), ('Comparison-group before-after', float(before_after['irr'])), ('Overlap observed-period average', float(overlap.iloc[0]))]
    mu_by = month12.groupby('segment').mu0.sum().reindex(ids).to_numpy()
    for label, irr in values:
        x = static.assign(benefit=mu_by * (1 - irr))
        x['ce'] = x.benefit / x.cost_yuan * 10000
        rr = ranked(x)
        row = compare_ranks(ranks['three'], rr, label)
        row.update({'common_IRR': irr, 'predicted_benefit': float(x.benefit.sum()), 'ce': float(x.benefit.sum() / cost.sum() * 10000)})
        scenarios.append(row)
    save_csv('effect assumption scenarios.csv', pd.DataFrame(scenarios))
    dump('effect assumption scenarios.json', scenarios)
    budgets = []
    selected_rows = []
    for fraction in [0.25, 0.5]:
        cap = float(cost.sum() * fraction)
        for kind, rank in ranks.items():
            spent = 0.0
            selected = []
            for row in rank.itertuples():
                if spent + row.cost_yuan <= cap + 1e-08:
                    selected.append(row.segment)
                    spent += row.cost_yuan
                    selected_rows.append({'budget_fraction': fraction, 'ranking': kind, 'segment': row.segment, 'priority_rank': row.rank, 'cost_yuan': row.cost_yuan, 'benefit': row.benefit})
            positions = [ids.index(s) for s in selected]
            draw = sim12[:, positions].sum(axis=1)
            benefit = float(seg.set_index('segment').loc[selected, 'benefit'].sum())
            lo, hi = interval(draw)
            budgets.append({'budget_fraction': fraction, 'ranking': kind, 'budget_yuan': cap, 'spent_yuan': spent, 'remaining_yuan': cap - spent, 'selected_segments': len(selected), 'benefit': benefit, 'benefit_ci_low': lo, 'benefit_ci_high': hi, 'share_of_all_predicted_benefit': benefit / summary12['benefit']})
    save_csv('budget scenarios.csv', pd.DataFrame(budgets))
    save_csv('budget selected segments.csv', pd.DataFrame(selected_rows))
    controls = p[p.treated.eq(0)]
    annual = controls.groupby(['year', 'segment']).month.nunique().unstack('segment')
    complete_years = annual.index[annual.eq(12).all(axis=1)]
    if len(complete_years) == 0:
        raise ValueError('Candidate analysis requires a complete calendar year for all controls.')
    reference_year = REFERENCE_YEAR or int(complete_years.max())
    if reference_year not in complete_years:
        raise ValueError('The reference year is incomplete for control segments.')
    candidate = controls[controls.year.eq(reference_year)].copy()
    assert len(candidate) == 12 * candidate.segment.nunique() and candidate.D.eq(0).all()
    candidate['D'] = 1
    candidate['time_after'] = candidate.month - 1
    cids = sorted(candidate.segment.unique())
    cb, cs, cm, _ = benefit_data(r, candidate, cids, draws)
    cr = b.set_index('segment').loc[cids, ['baseline_risk']].reset_index()
    cr['predicted_annual_without_treatment'] = cm.groupby('segment').mu0.sum().reindex(cids).to_numpy()
    cr['potential_annual_benefit'] = cb
    cr['benefit_ci_low'] = np.quantile(cs, 0.025, axis=0)
    cr['benefit_ci_high'] = np.quantile(cs, 0.975, axis=0)
    cr = cr.sort_values(['potential_annual_benefit', 'segment'], ascending=[False, True]).reset_index(drop=True)
    cr['screening_rank'] = np.arange(1, len(cids) + 1)
    save_csv('candidate screening.csv', cr)
    clo, chi = interval(cs.sum(axis=1))
    candidates = {'segments': len(cids), 'reference_year': reference_year, 'predicted_annual_crashes_without_treatment': float(cr.predicted_annual_without_treatment.sum()), 'scenario_annual_benefit': float(cb.sum()), 'benefit_ci_low': clo, 'benefit_ci_high': chi, 'rank_by': 'potential_annual_benefit', 'costs_used': False, 'top10_share_of_scenario_benefit': float(cr.potential_annual_benefit.head(10).sum() / cb.sum())}
    dump('candidate summary.json', candidates)
    speed = speed_analysis(raw_b, raw_m)
    np.savez_compressed(OUT / 'conditional simulation draws.npz', effect_coefficients=draws, treated_segment_ids=np.asarray(ids), benefit_12months=sim12, benefit_observed=sim_all, candidate_segment_ids=np.asarray(cids), candidate_benefit=cs)
    final = {'primary_model': point, 'observed_economics': summary_all, 'first12_economics': summary12, 'ranking_comparisons': comparisons, 'effect_assumption_scenarios': scenarios, 'top10_treated': ranks['three'].head(10).to_dict('records'), 'budget_scenarios': budgets, 'candidate_summary': candidates, 'top10_candidates': cr.head(10).to_dict('records'), 'speed': speed}
    dump('all results.json', final)
    after = {'input': sha(SOURCE)}
    assert hashes == after
    dump('calculation audit.json', {'input': SOURCE, 'input_sha256': hashes['input'], 'input_unchanged': True, 'coefficient_max_abs_difference': float(np.max(np.abs(r.params - saved_beta.reindex(r.params.index)))), 'covariance_max_abs_difference': float(np.max(np.abs(r.cov_params().to_numpy() - saved_cov.loc[r.params.index, r.params.index].to_numpy()))), 'n_annual_treated_months': len(first), 'n_annual_treated_segments': len(ids), 'all_annual_segment_month_counts_equal_12': True, 'candidate_costs_used': False, 'alpha': meta['alpha'], 'elapsed_seconds': time.time() - started, 'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': __import__('scipy').__version__, 'statsmodels': __import__('statsmodels').__version__}})
if __name__ == '__main__':
    main()
