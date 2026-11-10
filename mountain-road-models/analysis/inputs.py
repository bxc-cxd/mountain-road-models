from pathlib import Path
import hashlib
import os
import numpy as np
import pandas as pd
FIELDS=['aadt','length_used','width_used','low_road_class','measure_count']

def indicator_inputs(workbook,segments,required=True):
    path=os.environ.get('MICRO_INDICATOR_INPUTS','')
    if path:
        frame=pd.read_csv(Path(path))
    else:
        with pd.ExcelFile(workbook) as book:
            if '指标计算输入' not in book.sheet_names:
                if required:raise ValueError('Add the 指标计算输入 sheet or provide --indicator-inputs.')
                return None
            frame=pd.read_excel(book,sheet_name='指标计算输入')
    missing=set(['segment',*FIELDS])-set(frame.columns)
    if missing:raise ValueError('Missing indicator input columns: '+', '.join(sorted(missing)))
    frame=frame[['segment',*FIELDS]].copy()
    frame['segment']=frame['segment'].astype(str)
    frame=frame.set_index('segment',verify_integrity=True)
    ids=pd.Index(segments).astype(str)
    if not ids.is_unique or set(frame.index)!=set(ids):
        raise ValueError('Indicator inputs must contain each base-sheet segment exactly once.')
    frame=frame.loc[ids]
    frame[FIELDS]=frame[FIELDS].apply(pd.to_numeric)
    if not np.isfinite(frame[FIELDS].to_numpy(float)).all():raise ValueError('Indicator inputs must be finite.')
    if frame[['aadt','length_used','width_used']].le(0).any().any():raise ValueError('Traffic, length and width must be positive.')
    if not frame.low_road_class.isin([0,1]).all() or (frame.measure_count<0).any():raise ValueError('Invalid road-class indicator or measure count.')
    return frame

def risk_arguments(frame):
    return [frame[name].to_numpy(float) for name in FIELDS]

def indicator_digest(frame):
    return hashlib.sha256(frame.to_csv().encode('utf-8')).hexdigest()
