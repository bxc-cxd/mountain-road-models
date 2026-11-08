from pathlib import Path
import os,sys,json,time,warnings
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np,pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit
from scipy.linalg import LinAlgWarning,qr
from concurrent.futures import ProcessPoolExecutor,as_completed
from .bootstrap import irls
warnings.filterwarnings('ignore',category=LinAlgWarning)
from .config import folder, BOOTSTRAP_REPS as REPS, WORKERS
WORK=folder('cache'); OUT=folder('models')
Z=None; SEED=20261106

def initialize():
    global Z
    Z=dict(np.load(WORK/'staggered_design.npz'));Z['sid']=Z['sid'].astype(int)
    Z['ti']=np.flatnonzero(Z['treated']==1);Z['ci']=np.flatnonzero(Z['treated']==0)

def stable_prediction(z,w):
    pre=z['D']==0;post=~pre
    cy=np.bincount(z['cohort_code'][pre],weights=w[pre]*z['y'][pre],minlength=int(z['cohort_code'].max())+1)
    ty=np.bincount(z['calendar_code'][pre],weights=w[pre]*z['y'][pre],minlength=int(z['calendar_code'].max())+1)
    boundary=(cy[z['cohort_code']]==0)|(ty[z['calendar_code']]==0)
    keep=pre&(w>0)&~boundary
    X=z['X'];XX=X[keep]
    _,R,piv=qr(XX,mode='economic',pivoting=True)
    rank=int(np.sum(abs(np.diag(R))>abs(R[0,0])*1e-10));cols=piv[:rank]
    XX=XX[:,cols]
    start=np.linalg.lstsq(XX,X[keep]@z['beta'],rcond=1e-10)[0]
    beta,_=irls(XX,z['y'][keep],z['offset'][keep],w[keep],start=start)
    mu=np.zeros(int(post.sum()));active=(~boundary[post])&(w[post]>0)
    mu[active]=np.exp(z['offset'][post][active]+X[post][active][:,cols]@beta)
    reconstruction=np.linalg.lstsq(XX,X[keep],rcond=1e-10)[0]
    err=np.max(abs(X[post][active][:,cols]@reconstruction-X[post][active])) if active.any() else 0
    if err>1e-7:raise RuntimeError('out-of-sample mean has unidentified nuisance component')
    return mu

def draw(i):
    try:
        rng=np.random.default_rng(np.random.SeedSequence([SEED,i]))
        ids=np.r_[rng.choice(Z['ti'],len(Z['ti']),replace=True),rng.choice(Z['ci'],len(Z['ci']),replace=True)]
        mult=np.bincount(ids,minlength=len(Z['treated']));sid=Z['sid']
        bps,_=irls(Z['ps_X'],Z['treated'],np.zeros(len(mult)),mult,logistic=True)
        e=expit(Z['ps_X']@bps)
        if np.any((e<1e-12)|(e>1-1e-12)):raise RuntimeError('propensity at numerical boundary')
        w=np.where(Z['treated']==1,1,e/(1-e))[sid]*mult[sid]
        pre=Z['D']==0;post=~pre
        mu=stable_prediction(Z,w)
        estimate=float(np.sum(Z['y'][post]*mult[sid[post]])/np.sum(mu*mult[sid[post]]))
        return {'draw':i,'ok':True,'ratio':estimate}
    except Exception as ex:return {'draw':i,'ok':False,'error':type(ex).__name__+': '+str(ex)}

def main():
    from .config import require_inputs
    require_inputs('staggered')
    p=pd.read_pickle(WORK/'panel.pkl');baseline=pd.read_pickle(WORK/'baseline.pkl')
    p['cohort']=p.effective_date.dt.strftime('%Y-%m').fillna('never').astype(object)
    form='1 + C(cohort, Treatment(reference="never")) + C(calendar) + log_aadt_z + log_length_z + lane_width_z + speed_baseline_z + maintenance_baseline_z + baseline_risk_z + C(road_class)'
    Xm=patsy.dmatrix(form,p,return_type='dataframe');X=np.asarray(Xm);pre=p.D.eq(0).to_numpy();post=~pre
    m=sm.GLM(p.y[pre],X[pre],family=sm.families.Poisson(),offset=p.log_exposure[pre],freq_weights=p.att_weight[pre])
    r=m.fit(maxiter=300,tol=1e-10);assert r.converged
    beta,_=irls(X[pre],p.y.to_numpy()[pre],p.log_exposure.to_numpy()[pre],p.att_weight.to_numpy()[pre],start=np.asarray(r.params))
    mu=np.exp(p.log_exposure.to_numpy()[post]+X[post]@beta)
    ratio=float(p.y[post].sum()/mu.sum())
    postdata=p.loc[post,['segment','cohort','calendar','y','exposure']].copy();postdata['expected_untreated']=mu
    cells=postdata.groupby(['cohort','calendar']).agg(segments=('segment','nunique'),observed=('y','sum'),expected=('expected_untreated','sum'),exposure=('exposure','sum')).reset_index()
    cells['effect_count_per_segment']=(cells.observed-cells.expected)/cells.segments
    cells['ratio']=np.where(cells.expected>1e-8,cells.observed/cells.expected,np.nan)
    cells.to_csv(OUT/'cohort_time_effects.csv',index=False)
    (OUT/'cohort_time_coefficients.csv').write_text(pd.Series(beta,index=Xm.columns).to_csv(),encoding='utf-8')
    original=dict(np.load(WORK/'bootstrap_design.npz'))
    np.savez_compressed(WORK/'staggered_design.npz',X=X,beta=beta,y=p.y.to_numpy(),offset=p.log_exposure.to_numpy(),D=p.D.to_numpy(),
        cohort_code=pd.Categorical(p.cohort).codes.astype(int),calendar_code=pd.Categorical(p.calendar).codes.astype(int),
        sid=pd.Categorical(p.segment,categories=baseline.segment).codes,ps_X=original['ps_X'],treated=baseline.treated.to_numpy())
    cohortpre=p.loc[p.treated.eq(1)&p.D.eq(0)].groupby('cohort').y.sum()
    metadata={'model':'ATT-weighted cohort-time Poisson imputation','conditional_mean_formula':form,'calibration':'D=0, never-treated and pre-intervention treated observations',
       'n_obs_calibration':int(pre.sum()),'segments':int(p.segment.nunique()),'cohorts':int(p.loc[p.treated.eq(1),'cohort'].nunique()),'cohort_time_post_cells':len(cells),
       'zero_crash_post_cells':int(cells.observed.eq(0).sum()),'cohorts_with_zero_pre_crashes':cohortpre[cohortpre.eq(0)].index.tolist(),
       'observed_post':int(p.y[post].sum()),'expected_post':float(mu.sum()),'ratio':ratio,'aggregation':f'sum observed / sum counterfactual counts over {int(post.sum())} observed treated post segment-months',
       'covariate_slopes':'common across cohort and calendar time','cohort_definition':'exact effective calendar month','converged':bool(r.converged),
       'zero_baseline_cohorts':'Poisson boundary fitted mean approaches zero; individual ratios with zero predicted denominator are not reported.',
       'source':'Wooldridge (2023), Simple approaches to nonlinear difference-in-differences with panel data, doi:10.1093/ectj/utad016'}
    (OUT/'cohort_time_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print('point',metadata,flush=True)
    initialize();mcheck=stable_prediction(Z,p.att_weight.to_numpy())
    assert np.max(abs(mu-mcheck))<1e-7
    t0=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=initialize) as pool:
        for f in as_completed([pool.submit(draw,i) for i in range(REPS)]):
            rows.append(f.result())
            if len(rows)%500==0:print('completed',len(rows),'valid',sum(z['ok'] for z in rows),'seconds',round(time.time()-t0,1),flush=True)
    pd.DataFrame(rows).sort_values('draw').to_csv(OUT/'cohort_time_bootstrap.csv',index=False)
    valid=np.array([z['ratio'] for z in rows if z['ok']]);metadata.update(bootstrap_attempts=REPS,bootstrap_valid=len(valid),ci_low=float(np.quantile(valid,.025)),ci_high=float(np.quantile(valid,.975)),
        bootstrap_seed=SEED,bootstrap_seconds=time.time()-t0,failures=[z for z in rows if not z['ok']],bootstrap='stratified segment resampling, refit propensity model, ATT weights and untreated Poisson mean; percentile interval')
    (OUT/'cohort_time_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print('DONE',metadata,flush=True)

if __name__=='__main__':main()
