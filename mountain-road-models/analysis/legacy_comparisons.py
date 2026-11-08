import math,json
import numpy as np
import pandas as pd
from scipy.stats import norm,chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from . import core, models
from .config import folder
Z975=norm.ppf(.975)
OUT=folder('models')
def result_row(name: str, result, term: str, *, scale: str = "irr", note: str = "") -> dict:
    beta = float(result.params[term])
    se = float(result.bse[term])
    low = beta - Z975 * se
    high = beta + Z975 * se
    if scale == "irr":
        estimate, ci_low, ci_high = math.exp(beta), math.exp(low), math.exp(high)
    else:
        estimate, ci_low, ci_high = beta, low, high
    return {
        "analysis": name,
        "term": term,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coefficient": beta,
        "std_error": se,
        "p_value": float(result.pvalues[term]),
        "n_obs": int(result.nobs),
        "converged": bool(getattr(result, "converged", True)),
        "note": note,
    }

def positive_segment_panel(panel: pd.DataFrame) -> pd.DataFrame:
    totals = panel.groupby("segment")["y"].sum()
    return panel.loc[panel["segment"].isin(totals[totals.gt(0)].index)].copy()

def fit_cohort_interacted_ppml(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    data = positive_segment_panel(panel)
    data["cohort"] = data["effective_date"].dt.to_period("Q").astype(str)
    treated_base = data.loc[data["treated"].eq(1), ["segment", "cohort"]].drop_duplicates()
    cohort_counts = treated_base.groupby("cohort")["segment"].nunique().sort_index()
    terms = []
    term_to_cohort = {}
    for cohort in cohort_counts.index:
        term = f"post_{cohort.replace('Q', '_q')}"
        data[term] = (data["cohort"].eq(cohort) & data["D"].eq(1)).astype(int)
        if data[term].sum() > 0:
            terms.append(term)
            term_to_cohort[term] = cohort
    formula = "y ~ " + " + ".join(terms) + " + C(segment) + C(calendar)"
    result = core.fit_poisson(formula, data)

    retained_counts = cohort_counts.loc[[term_to_cohort[t] for t in terms]].astype(float)
    weights = retained_counts / retained_counts.sum()
    beta = np.array([float(result.params[t]) for t in terms])
    cov = result.cov_params().loc[terms, terms].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    beta_bar = float(w @ beta)
    se_bar = float(np.sqrt(max(w @ cov @ w, 0.0)))
    z = beta_bar / se_bar
    p_value = float(2 * sm.stats.norm.cdf(-abs(z))) if hasattr(sm.stats, "norm") else float(2 * __import__("scipy").stats.norm.cdf(-abs(z)))
    summary = {
        "analysis": "Full-sample cohort-interacted FE PPML",
        "term": "treated-cohort weighted post effect",
        "estimate": math.exp(beta_bar),
        "ci_low": math.exp(beta_bar - Z975 * se_bar),
        "ci_high": math.exp(beta_bar + Z975 * se_bar),
        "coefficient": beta_bar,
        "std_error": se_bar,
        "p_value": p_value,
        "n_obs": int(result.nobs),
        "converged": bool(result.converged),
        "note": "Treatment cohorts defined by implementation quarter; cohort coefficients combined using treated-segment shares.",
    }
    details = []
    for term in terms:
        row = result_row("Cohort-specific FE PPML", result, term)
        row["cohort"] = term_to_cohort[term]
        row["treated_segments_in_cohort"] = int(cohort_counts[term_to_cohort[term]])
        details.append(row)
    return summary, pd.DataFrame(details)

def main():
    from .config import require_inputs
    require_inputs('legacy_comparisons')
    p=pd.read_pickle(folder('cache')/'panel.pkl'); baseline=pd.read_pickle(folder('cache')/'baseline.pkl')
    fe=positive_segment_panel(p); rows=[]
    r=core.fit_poisson('y ~ D + C(segment) + C(calendar)',fe)
    rows.append(result_row('Segment and calendar-month FE PPML',r,'D'))
    threshold=baseline.pre_crashes.quantile(.9)
    trim=fe[fe.segment.isin(baseline.loc[baseline.pre_crashes.le(threshold),'segment'])]
    rt=core.fit_poisson('y ~ D + C(segment) + C(calendar)',trim)
    rows.append(result_row('FE PPML excluding top common-pre crash decile',rt,'D'))
    placebo=fe.copy()
    placebo['placebo_12m']=(placebo.treated.eq(1)&placebo.event_time.ge(-12)&placebo.event_time.lt(0)).astype(int)
    rp=core.fit_poisson('y ~ D + placebo_12m + C(segment) + C(calendar)',placebo)
    rows.append(result_row('Placebo 12 months early, actual treatment retained',rp,'placebo_12m',scale='coefficient'))
    cohort,details=fit_cohort_interacted_ppml(p); rows.append(cohort)
    details.to_csv(OUT/'legacy_cohort_details.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'legacy_comparisons.csv',index=False)
    cut=float(baseline.baseline_risk.median()); p['high_risk']=p.baseline_risk.ge(cut).astype(int)
    alpha=json.loads((OUT/'run_metadata.json').read_text())['alpha']
    r=smf.gee(models.TREND+' + high_risk + treated:high_risk + D:high_risk + time_after:high_risk',
        groups='segment',data=p,family=sm.families.NegativeBinomial(alpha=alpha),
        cov_struct=sm.cov_struct.Independence(),offset=p.log_exposure,weights=p.overlap_weight).fit(maxiter=500)
    assert r.converged
    m=float(np.average(p.loc[p.D.eq(1),'time_after'],weights=p.loc[p.D.eq(1),'overlap_weight']))
    def contrast(label,weights):
        c=pd.Series(0.,index=r.params.index)
        for k,v in weights.items(): c[k]=v
        b=float(c@r.params); se=float(np.sqrt(c@r.cov_params()@c))
        return dict(term=label,coefficient=b,std_error=se,estimate=float(np.exp(b)),
            ci_low=float(np.exp(b-Z975*se)),ci_high=float(np.exp(b+Z975*se)),p_value=float(2*norm.sf(abs(b/se))))
    hr=[contrast('D:high_risk',{'D:high_risk':1}),contrast('time_after:high_risk',{'time_after:high_risk':1}),
        contrast('Lower-risk effect at common mean post month',{'D':1,'time_after':m}),
        contrast('High-risk effect at common mean post month',{'D':1,'time_after':m,'D:high_risk':1,'time_after:high_risk':m}),
        contrast('High-risk/lower-risk IRR ratio at common mean post month',{'D:high_risk':1,'time_after:high_risk':m})]
    terms=['D:high_risk','time_after:high_risk']; b=r.params[terms].to_numpy(); V=r.cov_params().loc[terms,terms].to_numpy()
    w=float(b@np.linalg.solve(V,b))
    pd.DataFrame(hr).to_csv(OUT/'risk_interaction_results.csv',index=False)
    (OUT/'risk_interaction_metadata.json').write_text(json.dumps(dict(threshold=cut,common_post_month=m,
        weights='overlap',chi2=w,df=2,p_value=float(chi2.sf(w,2)),trim_threshold=float(threshold),
        retained_FE_segments=int(fe.segment.nunique()),trimmed_FE_segments=int(trim.segment.nunique())),indent=2))
    print('Fixed effects, placebo and risk interactions completed.',flush=True)
if __name__=='__main__':main()
