from pathlib import Path
import hashlib
import json
import os
DEPENDENCIES={'models':[],'event study':[],'comparisons':['models'],
              'bootstrap':['models'],'staggered':['models'],
              'economics':['models','comparisons'],'monetary':['economics'],
              'indicator sensitivity':[]}

def context(source):
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    settings={name:os.environ.get(name,'') for name in ['MICRO_EXCLUDE_YEAR','MICRO_REFERENCE_YEAR','MICRO_BOOTSTRAP_REPS']}
    for key in ['MICRO_INDICATOR_INPUTS','MICRO_VALUATION']:
        value=os.environ.get(key,'')
        settings[key]=hashlib.sha256(Path(value).read_bytes()).hexdigest() if value else None
    return {'input_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'analysis_sha256':digest.hexdigest(),'settings':settings}

def require_stages(source,output,stages):
    path=Path(output)/'pipeline state.json'
    if not path.exists():raise ValueError('Missing prerequisite stages: '+', '.join(stages))
    state=json.loads(path.read_text(encoding='utf-8'))
    if state.get('context')!=context(source):raise ValueError('Cached inputs or settings differ. Run the prerequisite stages.')
    missing=[name for name in stages if name not in state.get('completed',[])]
    if missing:raise ValueError('Missing prerequisite stages: '+', '.join(missing))

def begin_stage(source,output,stage):
    path=Path(output)/'pipeline state.json'
    current=context(source)
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if state.get('context')!=current:state={'context':current,'completed':[]}
    if DEPENDENCIES[stage]:require_stages(source,output,DEPENDENCIES[stage])
    invalid={stage}
    for _ in DEPENDENCIES:invalid.update(s for s,deps in DEPENDENCIES.items() if invalid.intersection(deps))
    state['completed']=[s for s in state['completed'] if s not in invalid]
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')

def complete_stage(output,stage):
    path=Path(output)/'pipeline state.json'
    state=json.loads(path.read_text(encoding='utf-8'))
    state['completed'].append(stage)
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')
