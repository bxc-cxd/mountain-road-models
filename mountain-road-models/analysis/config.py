from pathlib import Path
import os
INPUT = Path(os.environ.get('MICRO_INPUT', 'data/private_input.xlsx')).resolve()
OUTPUT = Path(os.environ.get('MICRO_OUTPUT', 'results')).resolve()
STUDY = os.environ.get('MICRO_MODE', 'study') == 'study'
BOOTSTRAP_REPS = int(os.environ.get('MICRO_BOOTSTRAP_REPS', '2000'))
WORKERS = int(os.environ.get('MICRO_WORKERS', '4'))
def folder(name):
    p=OUTPUT/name; p.mkdir(parents=True,exist_ok=True); return p

def require_inputs(stage):
    from .provenance import DEPENDENCIES, require_stages
    if DEPENDENCIES[stage]:
        require_stages(INPUT,OUTPUT,DEPENDENCIES[stage])
