import hashlib
import json
import numpy as np
import pandas as pd
from .config import INPUT,VALUATION,folder,require_inputs

def price_factor(year,target,cpi):
    year=int(year);target=int(target)
    years=range(min(year,target)+1,max(year,target)+1)
    missing=[y for y in years if str(y) not in cpi]
    if missing:raise ValueError('Missing CPI years: '+', '.join(map(str,missing)))
    value=float(np.prod([cpi[str(y)]/100 for y in years]))
    if not np.isfinite(value) or value<=0:raise ValueError('CPI factors must be positive.')
    return value if year<=target else 1/value

def main():
    require_inputs('monetary')
    out=folder('monetary');calc=folder('economics')
    parameters=json.loads(VALUATION.read_text(encoding='utf-8'))
    target=int(parameters['price_year']);base_year=int(parameters['unit_cost_year'])
    cpi=parameters['cpi_previous_year_100'];unit=parameters['unit_crash_costs']
    names=['property_damage_only','minor_injury','severe_injury','fatal']
    if not all(np.isfinite(unit[k]) and unit[k]>=0 for k in names):raise ValueError('Crash costs must be finite and nonnegative.')
    b=pd.read_excel(INPUT,sheet_name='路段基础数据');m=pd.read_excel(INPUT,sheet_name='月度面板数据')
    treated=b[b['处理组标记'].eq(1)].copy()
    pre=m[m['路段编号'].isin(treated['路段编号'])].copy()
    dates=pd.to_datetime(pre['日历月份'])
    starts=pre['路段编号'].map(pd.to_datetime(treated.set_index('路段编号')['干预开始日期']))
    pre=pre[dates.lt(starts.dt.to_period('M').dt.to_timestamp())]
    columns=['死亡事故数','伤人事故数','轻微事故数']
    if not pre[columns].sum(axis=1).eq(pre['月度总事故数_最终']).all():raise ValueError('Crash categories must sum to total crashes.')
    counts=pre[columns].sum().astype(int)
    if counts.sum()==0:raise ValueError('Preconstruction crashes are required for valuation weights.')
    cost=treated['工程费用_元'].astype(float)
    startyear=pd.to_datetime(treated['干预开始日期']).dt.year
    endyear=pd.to_datetime(treated['干预结束日期']).dt.year
    factors=startyear.map(lambda y:price_factor(y,target,cpi))
    cost_target=float((cost*factors).sum())
    completion_cost=float((cost*endyear.map(lambda y:price_factor(y,target,cpi))).sum())
    detail=pd.DataFrame({'segment':treated['路段编号'],'construction_start_year':startyear,
        'construction_completion_year':endyear,'recorded_cost_yuan':cost,
        'price_factor':factors,'price_year':target,'converted_cost_yuan':cost*factors})
    detail.to_csv(out/'cost price conversion.csv',index=False,encoding='utf-8-sig')
    factor=price_factor(base_year,target,cpi)
    adjusted={key:unit[key]*factor for key in names}
    draws=np.load(calc/'conditional simulation draws.npz')['benefit_12months'].sum(axis=1)
    econ=pd.read_csv(calc/'economic summary.csv').iloc[1]
    assert np.isclose(np.quantile(draws,.025),econ['benefit_ci_low'])
    scenarios=[]
    for label,minor,injury in [('Lower valuation','property_damage_only','minor_injury'),
                              ('Upper valuation','minor_injury','severe_injury')]:
        mean=(counts['死亡事故数']*adjusted['fatal']+counts['伤人事故数']*adjusted[injury]+counts['轻微事故数']*adjusted[minor])/counts.sum()
        benefit=float(econ['benefit']*mean);low,high=np.quantile(draws*mean/cost_target,[.025,.975])
        scenarios.append({'scenario':label,'minor_category_valuation':minor,'injury_category_valuation':injury,
            'price_year':target,'mean_crash_value_yuan':float(mean),'first_year_benefit_yuan':benefit,
            'first_year_benefit_to_initial_cost':benefit/cost_target,'conditional_ci_low':float(low),'conditional_ci_high':float(high)})
    result={'price_year':target,'unit_cost_year':base_year,'cpi_previous_year_100':cpi,
        'unit_price_factor':factor,'nominal_engineering_cost_yuan':float(cost.sum()),
        'engineering_cost_yuan':cost_target,'completion_year_cost_yuan':completion_cost,
        'preconstruction_segment_months':len(pre),'preconstruction_severity_counts':counts.to_dict(),
        'preconstruction_crashes':int(counts.sum()),'unit_crash_values':unit,'adjusted_unit_crash_values':adjusted,
        'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'scenarios':scenarios}
    (out/'monetary results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(scenarios).to_csv(out/'monetary results.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
