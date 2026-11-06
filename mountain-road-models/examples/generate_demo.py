from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.indicators import calculate_risk,calculate_maintenance
def main():
    ap=argparse.ArgumentParser(description='Generate wholly simulated data; no study data are read or sampled.');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():ap.error('Output already exists. Choose a new path to avoid overwriting a workbook.')
    rng=np.random.default_rng(733010); n=146; dates=pd.date_range('2021-01-01','2025-12-01',freq='MS')
    ids=[f'DEMO{i+1:04d}' for i in range(n)]
    treated=np.r_[np.ones(74,dtype=int),np.zeros(72,dtype=int)]
    aadt=rng.integers(1500,9000,n);length=rng.integers(200,1500,n);width=rng.choice([3.,3.5,3.75],n)
    cls=rng.choice(['二级','三级','四级'],n); flags=rng.integers(0,2,(n,6));flags[treated==0]=0
    risk=calculate_risk(aadt,length,width,cls!='二级',flags.sum(axis=1)+rng.integers(0,3,n))
    scores,_,_=calculate_maintenance(risk,len(dates))
    effective=[dates[int(k)] for k in rng.integers(18,42,74)]+[pd.NaT]*72
    start=[e-pd.DateOffset(months=int(rng.integers(1,4))) if pd.notna(e) else pd.NaT for e in effective]
    end=[e-pd.DateOffset(months=1) if pd.notna(e) else pd.NaT for e in effective]
    cost=np.where(treated==1,rng.uniform(10000,150000,n),0)
    b=pd.DataFrame({'路段编号':ids,'路线编号':[f'DEMOR{i//5:03d}' for i in range(n)],'原始道路等级':cls,
        '路段长度_米':length,'车道数':2,'单车道宽度_米':width,'年平均日交通量':aadt,
        '处理组标记':treated,'干预生效月份':effective,'干预开始日期':start,'干预结束日期':end,
        '工程费用_元':cost,'基础风险指数':risk,'治理组合强度':flags.sum(axis=1),
        '原始治理措施':['SIMULATED package' if t else None for t in treated]})
    for j,name in enumerate(['警示信息','视觉诱导','速度管理','路面排水','视距改善','路侧防护']):b[name]=flags[:,j]
    rows=[];segment_noise=rng.normal(0,.2,n); speed0=rng.uniform(40,65,n)
    for i,sid in enumerate(ids):
        for j,date in enumerate(dates):
            e=effective[i];event=(date.year-e.year)*12+date.month-e.month if pd.notna(e) else np.nan
            D=int(treated[i] and event>=0);h=max(0,event) if D else 0
            exposure=aadt[i]*length[i]/1000*date.days_in_month
            mu=exposure/1e6*.65*np.exp(.1*risk[i]+segment_noise[i]+.1*np.sin(j/6)-.35*D)
            y=int(rng.negative_binomial(1.5,1.5/(1.5+mu)))
            fatal,injury,minor=rng.multinomial(y,[.03,.3,.67])
            rows.append({'路段编号':sid,'年份':date.year,'月份':date.month,'日历月份':date,
                '干预后标记':D,'干预后月数':h,'事件时间_相对月份':event,'月度总事故数_最终':y,
                '月度交通暴露量':exposure,'第85百分位速度':speed0[i]+rng.normal(0,2)-.5*D,
                '养护评分':scores[i,j],'死亡事故数':fatal,'伤人事故数':injury,'轻微事故数':minor,
                '干预开始日期':start[i],'干预结束日期':end[i]})
    p=pd.DataFrame(rows);b['汇总总事故数']=b['路段编号'].map(p.groupby('路段编号')['月度总事故数_最终'].sum())
    with pd.ExcelWriter(a.output,engine='openpyxl') as w:
        b.to_excel(w,sheet_name='路段基础数据',index=False);p.to_excel(w,sheet_name='月度面板数据',index=False)
        pd.DataFrame({'NOTICE':['WHOLLY SIMULATED. NOT THE STUDY DATA. Outputs are demonstrations, not paper estimates.']}).to_excel(w,sheet_name='SIMULATED_ONLY',index=False)
    print(json.dumps({'output':str(a.output),'simulated_only':True,'segments':n,'rows':len(p),'seed':733010}))
if __name__=='__main__':main()
