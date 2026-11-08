from pathlib import Path
import sys, json, hashlib
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from .config import INPUT, STUDY, folder, require_inputs
require_inputs('monetary')
WORK=folder('monetary'); CALC=folder('economics')
expected=hashlib.sha256(INPUT.read_bytes()).hexdigest()
b=pd.read_excel(INPUT,sheet_name='路段基础数据');m=pd.read_excel(INPUT,sheet_name='月度面板数据')
t=b[b['处理组标记'].eq(1)].copy()
pre=m[m['路段编号'].isin(t['路段编号'])].copy()
pre=pre[pd.to_datetime(pre['日历月份']).lt(pd.to_datetime(pre['干预开始日期']).dt.to_period('M').dt.to_timestamp())]
cols=['死亡事故数','伤人事故数','轻微事故数']
assert pre[cols].sum(axis=1).eq(pre['月度总事故数_最终']).all()
counts=pre[cols].sum().astype(int)
if STUDY: assert int(counts.sum())==96
cpi={2018:102.1,2019:102.9,2020:102.5,2021:100.9,2022:102.0,2023:100.2,2024:100.2}
def inflate(year):return float(np.prod([cpi[y]/100 for y in range(int(year)+1,2025)]))
cost=t['工程费用_元'].astype(float)
startyear=pd.to_datetime(t['干预开始日期']).dt.year
endyear=pd.to_datetime(t['干预结束日期']).dt.year
cost2024=float((cost*startyear.map(inflate)).sum())
cost2024_completion=float((cost*endyear.map(inflate)).sum())
cost_detail=pd.DataFrame({'segment':t['路段编号'],'construction_start_year':startyear,
    'construction_completion_year':endyear,'recorded_cost_yuan':cost,
    'start_year_to_2024_factor':startyear.map(inflate),'cost_2024_yuan':cost*startyear.map(inflate)})
cost_detail.to_csv(WORK/'cost_price_conversion.csv',index=False,encoding='utf-8-sig')
unit2017={'property_damage_only':11274,'minor_injury':20223,'severe_injury':865544,'fatal':3181394}
unit2024={k:v*inflate(2017) for k,v in unit2017.items()}
raw=np.load(CALC/'conditional_simulation_draws.npz')
print('simulation keys',raw.files)
key='benefit_12months'
drawtot=raw[key].sum(axis=1)
econ=pd.read_csv(CALC/'economic_summary.csv').iloc[1]
benefit=float(econ['benefit'])
assert np.isclose(np.quantile(drawtot,.025),econ['benefit_ci_low'])
scenarios=[]
for label,minor,injury in [('Lower valuation','property_damage_only','minor_injury'),
                           ('Upper valuation','minor_injury','severe_injury')]:
    avg=(counts['死亡事故数']*unit2024['fatal']+counts['伤人事故数']*unit2024[injury]+counts['轻微事故数']*unit2024[minor])/counts.sum()
    monetary=benefit*avg;ratio=monetary/cost2024
    lo,hi=np.quantile(drawtot*avg/cost2024,[.025,.975])
    scenarios.append({'scenario':label,'minor_category_valuation':minor,'injury_category_valuation':injury,
        'mean_crash_value_2024_yuan':float(avg),'first_year_benefit_2024_yuan':float(monetary),
        'first_year_benefit_to_initial_cost':float(ratio),'conditional_ci_low':float(lo),'conditional_ci_high':float(hi)})
out={'source_workbook_sha256':expected,'price_year':2024,'construction_cost_year_fact':'Construction-year cost',
    'construction_cost_conversion':'construction-start year; completion-year sensitivity recorded separately',
    'cpi_previous_year_100':cpi,'2017_to_2024_factor':inflate(2017),'nominal_engineering_cost_yuan':float(cost.sum()),
    'engineering_cost_2024_yuan':cost2024,'completion_year_sensitivity_cost_2024_yuan':cost2024_completion,
    'preconstruction_segment_months':len(pre),'preconstruction_severity_counts':counts.to_dict(),
    'preconstruction_crashes':int(counts.sum()),'unit_crash_values_2017_yuan':unit2017,'unit_crash_values_2024_yuan':unit2024,
    'source':'Tan, H., Zhao, F., Hao, H., & Liu, Z. (2020). Cost analysis of road traffic crashes in China. International Journal of Injury Control and Safety Promotion, 27(3), 385–391. https://doi.org/10.1080/17457300.2020.1785507',
    'assumptions':'Pooled treated preconstruction severity composition held fixed; common proportional crash reduction across valuation categories; no estimated severity-specific effect; first 12 effective months versus one-off initial capital, excluding maintenance and later-life benefits.',
    'scenarios':scenarios}
(WORK/'monetary_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(scenarios).to_csv(WORK/'monetary_results.csv',index=False,encoding='utf-8-sig')
print(json.dumps(out,ensure_ascii=False,indent=2))
