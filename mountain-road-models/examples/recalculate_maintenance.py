from pathlib import Path
import argparse,sys,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from analysis.indicators import calculate_maintenance,MAINTENANCE_CEILING

def main():
    ap=argparse.ArgumentParser(description='Write scores from the current formula to a NEW CSV; never edit the workbook.')
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():ap.error('Output already exists; use a new CSV path.')
    before=hashlib.sha256(a.input.read_bytes()).hexdigest()
    base=pd.read_excel(a.input,sheet_name='路段基础数据')
    panel=pd.read_excel(a.input,sheet_name='月度面板数据')
    ids=base['路段编号'].astype(str).tolist()
    dates=sorted(pd.to_datetime(panel['日历月份']).dt.to_period('M').dt.to_timestamp().unique())
    scores,_,_=calculate_maintenance(base['基础风险指数'].to_numpy(float),len(dates))
    index=pd.MultiIndex.from_product([ids,dates],names=['路段编号','日历月份'])
    frame=pd.Series(scores.ravel(),index=index,name='recalculated_maintenance').reset_index()
    original=panel[['路段编号','日历月份','养护评分']].copy()
    original['日历月份']=pd.to_datetime(original['日历月份']).dt.to_period('M').dt.to_timestamp()
    frame=frame.merge(original,on=['路段编号','日历月份'],validate='one_to_one')
    frame['difference']=frame.recalculated_maintenance-frame['养护评分']
    assert len(frame)==len(panel)==len(index),'A balanced segment-month panel is required.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(a.output,index=False,encoding='utf-8-sig')
    assert before==hashlib.sha256(a.input.read_bytes()).hexdigest()
    print(f'Formula ceiling: {MAINTENANCE_CEILING:g}; changed monthly values: {frame.difference.ne(0).sum()}; input unchanged.')
if __name__=='__main__':main()
