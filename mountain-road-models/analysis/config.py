from pathlib import Path
import os
INPUT=Path(os.environ.get('MICRO_INPUT','data/input.xlsx')).resolve()
OUTPUT=Path(os.environ.get('MICRO_OUTPUT','results')).resolve()
VALUATION=Path(os.environ.get('MICRO_VALUATION',str(Path(__file__).resolve().parents[1]/'parameters/valuation.json'))).resolve()
EXCLUDE_YEAR=int(os.environ['MICRO_EXCLUDE_YEAR']) if os.environ.get('MICRO_EXCLUDE_YEAR') else None
REFERENCE_YEAR=int(os.environ['MICRO_REFERENCE_YEAR']) if os.environ.get('MICRO_REFERENCE_YEAR') else None
BOOTSTRAP_REPS=int(os.environ.get('MICRO_BOOTSTRAP_REPS','2000'))
WORKERS=int(os.environ.get('MICRO_WORKERS','4'))

def folder(name):
    path=OUTPUT/name
    path.mkdir(parents=True,exist_ok=True)
    return path

def require_inputs(stage):
    from .provenance import DEPENDENCIES,require_stages
    if DEPENDENCIES[stage]:require_stages(INPUT,OUTPUT,DEPENDENCIES[stage])
