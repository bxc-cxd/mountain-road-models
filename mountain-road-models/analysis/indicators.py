import numpy as np
SEED = 20260529
NORMAL_OFFSET = 9743
def zscore(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all() or x.std(ddof=0)==0:
        raise ValueError('Standardization requires finite, varying inputs.')
    return (x-x.mean())/x.std(ddof=0)
def calculate_risk(aadt, construction_length, construction_width, low_road_class, measure_count,
                   *, weights=(.38, .33, .20, .16, .12), noise_scale=.45):
    epsilon=np.random.default_rng(SEED).standard_normal(len(aadt))
    score=(weights[0]*zscore(np.log1p(aadt))+weights[1]*zscore(np.log1p(construction_length))
           +weights[2]*zscore(1/np.asarray(construction_width))+weights[3]*zscore(low_road_class)
           +weights[4]*zscore(measure_count)+noise_scale*epsilon)
    return np.round(zscore(score),4)
def calculate_maintenance(risks, months=60, *, intercept=82.0, risk_slope=-4.5, noise_scale=5.0):
    risks=np.asarray(risks,dtype=float)
    if risks.ndim!=1 or not np.isfinite(risks).all():
        raise ValueError('Provide a finite risk vector in construction segment order.')
    if not isinstance(months,(int,np.integer)) or months<1:
        raise ValueError('months must be a positive integer.')
    n=len(risks)
    noise=np.random.default_rng(SEED).standard_normal(NORMAL_OFFSET+n*months)[NORMAL_OFFSET:].reshape(n,months)
    latent=intercept+risk_slope*risks[:,None]+noise_scale*noise
    return np.round(np.minimum(100.0,latent),1),latent,noise

def maintenance_for_panel(segments, risks, panel_segments, panel_dates, **parameters):
    import pandas as pd
    segments=pd.Index(segments).astype(str)
    dates=pd.DatetimeIndex(panel_dates)
    months=dates.unique().sort_values()
    expected_months=pd.date_range(months.min(),months.max(),freq='MS')
    if not segments.is_unique or len(segments)!=len(risks):
        raise ValueError('Provide one risk score per unique segment in construction order.')
    if not months.equals(expected_months):
        raise ValueError('Provide the complete monthly calendar before filtering observations.')
    canonical=pd.MultiIndex.from_product([segments,months])
    observed=pd.MultiIndex.from_arrays([pd.Index(panel_segments).astype(str),dates])
    positions=canonical.get_indexer(observed)
    if not observed.is_unique or len(observed)!=len(canonical) or (positions<0).any():
        raise ValueError('Provide the complete balanced segment-month panel before filtering or resampling.')
    scores,_,_=calculate_maintenance(risks,len(months),**parameters)
    return scores.ravel()[positions]
