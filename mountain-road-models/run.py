from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from analysis.provenance import DEPENDENCIES,begin_stage,complete_stage
STAGES=list(DEPENDENCIES)

def main():
    parser=argparse.ArgumentParser(description='Run segment-panel analyses.')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',nargs='+',choices=STAGES,default=STAGES)
    parser.add_argument('--indicator-inputs',type=Path)
    parser.add_argument('--valuation',type=Path,default=Path(__file__).resolve().parent/'parameters/valuation.json')
    parser.add_argument('--exclude-year',type=int)
    parser.add_argument('--reference-year',type=int)
    parser.add_argument('--bootstrap-reps',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--verbose',action='store_true')
    args=parser.parse_args()
    src=args.input.resolve();out=args.output.resolve()
    if not src.is_file():parser.error('Input workbook not found.')
    if args.bootstrap_reps<2 or args.workers<1:parser.error('Require bootstrap-reps >= 2 and workers >= 1.')
    if src==out or src.is_relative_to(out):parser.error('Place the input outside the output directory.')
    if args.indicator_inputs and not args.indicator_inputs.is_file():parser.error('Indicator-input CSV not found.')
    if not args.valuation.is_file():parser.error('Valuation parameters not found.')
    order=[]
    def add(stage):
        for dependency in DEPENDENCIES[stage]:add(dependency)
        if stage not in order:order.append(stage)
    for stage in args.steps:add(stage)
    env=os.environ.copy()
    env.update(MICRO_INPUT=str(src),MICRO_OUTPUT=str(out),
               MICRO_INDICATOR_INPUTS=str(args.indicator_inputs.resolve()) if args.indicator_inputs else '',
               MICRO_VALUATION=str(args.valuation.resolve()),
               MICRO_EXCLUDE_YEAR=str(args.exclude_year) if args.exclude_year is not None else '',
               MICRO_REFERENCE_YEAR=str(args.reference_year) if args.reference_year is not None else '',
               MICRO_BOOTSTRAP_REPS=str(args.bootstrap_reps),MICRO_WORKERS=str(args.workers),
               PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    os.environ.update(env)
    if 'indicator sensitivity' in order:
        import pandas as pd
        from analysis.inputs import indicator_inputs
        base=pd.read_excel(src,sheet_name='路段基础数据')
        indicator_inputs(src,base['路段编号'].astype(str),required=True)
    out.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(src.read_bytes()).hexdigest();started=time.time()
    for stage in order:
        begin_stage(src,out,stage)
        if args.verbose:print(stage,flush=True)
        subprocess.run([sys.executable,'-m','analysis.'+stage],cwd=Path(__file__).resolve().parent,
                       env=env,check=True,stdout=None if args.verbose else subprocess.DEVNULL)
        complete_stage(out,stage)
    if hashlib.sha256(src.read_bytes()).hexdigest()!=before:raise RuntimeError('Input workbook changed during execution.')
    (out/'run manifest.json').write_text(json.dumps({'input_sha256':before,'completed':order,
        'bootstrap_reps':args.bootstrap_reps,'workers':args.workers,
        'elapsed_seconds':time.time()-started,'python':sys.version},indent=2),encoding='utf-8')

if __name__=='__main__':main()
