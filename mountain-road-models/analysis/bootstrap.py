from pathlib import Path
import os,sys,json,time,csv
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.special import expit,gammaln,digamma
from scipy.optimize import minimize
from scipy.linalg import solve
from concurrent.futures import ProcessPoolExecutor,as_completed

from .config import folder, BOOTSTRAP_REPS as REPS, WORKERS
WORK=folder('cache'); OUT=folder('models')
SEED=20261105
Z=None

def initialize():
    global Z
    Z=dict(np.load(WORK/'bootstrap_design.npz'))
    Z['sid']=Z['sid'].astype(int)
    Z['ti']=np.flatnonzero(Z['treated']==1);Z['ci']=np.flatnonzero(Z['treated']==0)

def irls(X,y,off,w,a=0,start=None,logistic=False):
    b=np.zeros(X.shape[1]) if start is None else start.copy()
    for k in range(150):
        eta=X@b+off
        if logistic:
            mu=expit(eta);W=w*np.maximum(mu*(1-mu),1e-12);sc=X.T@(w*(y-mu))
        else:
            mu=np.exp(np.clip(eta,-50,50));W=w*mu/(1+a*mu);sc=X.T@(w*(y-mu)/(1+a*mu))
        H=X.T@(W[:,None]*X)
        try:step=solve(H,sc,assume_a='pos')
        except Exception:step=np.linalg.lstsq(H,sc,rcond=1e-12)[0]
        if max(abs(step))>5:step=step*5/max(abs(step))
        b+=step
        if max(abs(step))<1e-9 or max(abs(sc))<1e-9:break
    if k==149 or not np.isfinite(b).all():raise RuntimeError('IRLS did not converge')
    if logistic and max(abs(b))>70:raise RuntimeError('propensity separation')
    return b,k+1

def dispersion(mult):
    X=Z['Xalpha'];y=Z['y'];off=Z['offset'];w=mult[Z['sid']];scale=w.sum()
    def fg(z):
        eta=off+X@z[:-1];mu=np.exp(eta);a=np.exp(z[-1]);rr=1/a
        ll=gammaln(y+rr)-gammaln(rr)-gammaln(y+1)-rr*np.log1p(a*mu)+y*(np.log(a)+eta-np.log1p(a*mu))
        gb=X.T@(w*(y-mu)/(1+a*mu))
        dr=digamma(y+rr)-digamma(rr)-np.log1p(a*mu)+1-(rr+y)/(rr+mu)
        return -float(w@ll)/scale,-np.r_[gb,np.sum(w*(-rr*dr))]/scale
    z0=np.r_[Z['alpha_beta'],np.log(float(Z['alpha']))]
    opt=minimize(fg,z0,method='L-BFGS-B',jac=True,bounds=[(None,None)]*X.shape[1]+[(np.log(1e-5),np.log(100))],
        options={'maxiter':1000,'ftol':2e-12,'gtol':1e-8,'maxls':40})
    if not opt.success or max(abs(opt.jac[:-1]))>1e-5:raise RuntimeError('dispersion fit: '+str(opt.message))
    a=float(np.exp(opt.x[-1]));boundary=a<1.01e-5
    return (0.0 if boundary else a),boundary,int(opt.nit)

def evaluate(mult,index):
    try:
        Xp=Z['ps_X'];T=Z['treated'];sid=Z['sid']
        bp,_=irls(Xp,T,np.zeros(len(T)),mult,logistic=True)
        e=expit(Xp@bp)
        if np.any((e<1e-12)|(e>1-1e-12)):raise RuntimeError('propensity numerically at boundary')
        rw=np.where(T==1,1,e/(1-e));rw/=np.average(rw,weights=mult)
        a,boundary,nit=dispersion(mult)
        beta,iterations=irls(Z['X'],Z['y'],Z['offset'],mult[sid]*rw[sid],a,start=Z['beta'])
        post=Z['D']==1;w=mult[sid[post]]*rw[sid[post]]
        mean=float(np.average(Z['time_after'][post],weights=w))
        theta=float(beta[int(Z['D_index'])]+mean*beta[int(Z['trend_index'])])
        return {'draw':index,'ok':True,'theta':theta,'irr':float(np.exp(theta)),'alpha':a,'alpha_boundary':boundary,'mean_post_month':mean,
           'max_raw_control_weight':float((e[T==0]/(1-e[T==0])).max()),'alpha_iterations':nit,'gee_iterations':iterations,'error':''}
    except Exception as ex:
        return {'draw':index,'ok':False,'error':type(ex).__name__+': '+str(ex)}

def replicate(index):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,index]))
    ids=np.r_[rng.choice(Z['ti'],len(Z['ti']),replace=True),rng.choice(Z['ci'],len(Z['ci']),replace=True)]
    return evaluate(np.bincount(ids,minlength=len(Z['treated'])),index)

def main():
    from .config import require_inputs
    require_inputs('bootstrap')
    initialize();t0=time.time();check=evaluate(np.ones(len(Z['treated'])),-1)
    expected=float(Z['beta'][int(Z['D_index'])]+np.mean(Z['time_after'][Z['D']==1])*Z['beta'][int(Z['trend_index'])])
    check['original_theta']=expected;check['theta_difference']=check.get('theta',float('nan'))-expected
    print('point-validation',check,flush=True)
    (OUT/'bootstrap_point_validation.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    assert check['ok'] and abs(check['theta_difference'])<2e-5
    count=REPS
    rows=[]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=initialize) as pool:
        futures=[pool.submit(replicate,i) for i in range(count)]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%100==0:
                print('completed',len(rows),'valid',sum(r['ok'] for r in rows),'seconds',round(time.time()-t0,1),flush=True)
                (OUT/'bootstrap_progress.json').write_text(json.dumps({'completed':len(rows),'valid':sum(r['ok'] for r in rows),'seconds':time.time()-t0}),encoding='utf-8')
    rows.sort(key=lambda r:r['draw']);fields=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/'bootstrap_draws.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fields);wr.writeheader();wr.writerows(rows)
    valid=[r for r in rows if r['ok']];irrs=np.array([r['irr'] for r in valid]);thetas=np.array([r['theta'] for r in valid])
    result={'replicates_attempted':count,'replicates_valid':len(valid),'seed':SEED,'stratification':f"treatment group ({len(Z['ti'])} treated, {len(Z['ci'])} controls)",
        'sampling_unit':'complete road segment','estimated_components':['propensity model','ATT weights','pilot NB2 dispersion','NB-GEE coefficients','observed post-period averaging month'],
        'point_estimate':float(np.exp(expected)),'ci_low':float(np.quantile(irrs,.025)),'ci_high':float(np.quantile(irrs,.975)),
        'bootstrap_se_log_irr':float(thetas.std(ddof=1)),'median_irr':float(np.median(irrs)),'alpha_boundary_fits':sum(r['alpha_boundary'] for r in valid),
        'failures':[r for r in rows if not r['ok']],'seconds':time.time()-t0,'interval':'percentile, no bootstrap p-value reported'}
    (OUT/'bootstrap_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('DONE',result,flush=True)

if __name__=='__main__':main()
