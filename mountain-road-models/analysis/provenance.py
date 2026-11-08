from pathlib import Path
import hashlib
import json

DEPENDENCIES = {
    'models': [], 'event_study': [], 'legacy_comparisons': ['models'],
    'bootstrap': ['models'], 'staggered': ['models'], 'economics': ['models'],
    'monetary': ['economics'], 'indicator_sensitivity': [],
}

def context(source):
    root=Path(__file__).resolve().parent
    digest=hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return {'input_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'analysis_sha256':digest.hexdigest()}

def require_stages(source, output, stages):
    path=Path(output)/'pipeline_state.json'
    if not path.exists():
        raise ValueError('Run the prerequisite stages with run.py: '+', '.join(stages))
    state=json.loads(path.read_text(encoding='utf-8'))
    if state.get('context')!=context(source):
        raise ValueError('Cached outputs use a different input or code version. Rerun the model pipeline.')
    missing=[s for s in stages if s not in state.get('completed',[])]
    if missing:
        raise ValueError('Required stages have not completed: '+', '.join(missing))

def begin_stage(source, output, stage):
    path=Path(output)/'pipeline_state.json'
    current=context(source)
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if state.get('context')!=current:
        state={'context':current,'completed':[]}
    if DEPENDENCIES[stage]:
        require_stages(source,output,DEPENDENCIES[stage])
    invalid={stage}
    for _ in DEPENDENCIES:
        invalid.update(s for s,deps in DEPENDENCIES.items() if invalid.intersection(deps))
    state['completed']=[s for s in state['completed'] if s not in invalid]
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')

def complete_stage(output, stage):
    path=Path(output)/'pipeline_state.json'
    state=json.loads(path.read_text(encoding='utf-8'))
    state['completed'].append(stage)
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')
