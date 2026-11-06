import numpy as np
MAINTENANCE_CEILING = 100.0
SEED = 20260529
NORMAL_OFFSET = 9743
def zscore(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all() or x.std(ddof=0)==0:
        raise ValueError('Standardization requires finite, varying inputs.')
    return (x-x.mean())/x.std(ddof=0)
def calculate_risk(aadt, construction_length, construction_width, low_road_class, measure_count):
    epsilon=np.random.default_rng(SEED).standard_normal(len(aadt))
    score=(.38*zscore(np.log1p(aadt))+.33*zscore(np.log1p(construction_length))
           +.20*zscore(1/np.asarray(construction_width))+.16*zscore(low_road_class)
           +.12*zscore(measure_count)+.45*epsilon)
    return np.round(zscore(score),4)
def calculate_maintenance(risks, months=60):
    risks=np.asarray(risks,dtype=float)
    if risks.ndim!=1 or not np.isfinite(risks).all():
        raise ValueError('Provide a finite risk vector in construction segment order.')
    n=len(risks)
    noise=np.random.default_rng(SEED).standard_normal(NORMAL_OFFSET+n*months)[NORMAL_OFFSET:].reshape(n,months)
    latent=82.0-4.5*risks[:,None]+5.0*noise
    return np.round(np.minimum(MAINTENANCE_CEILING,latent),1),latent,noise
