from pathlib import Path
import argparse,os,subprocess,sys,json,hashlib,time
STAGES=['models','event_study','legacy_comparisons','bootstrap','staggered','economics','monetary']
def main():
    ap=argparse.ArgumentParser(description='Run the analysis with a restricted workbook or a simulated example.')
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['study','demo'],default='study')
    ap.add_argument('--steps',nargs='+',choices=STAGES,default=STAGES)
    ap.add_argument('--bootstrap-reps',type=int,default=2000)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args(); src=a.input.resolve(); out=a.output.resolve()
    if not src.is_file(): ap.error(f'Input workbook not found: {src}')
    if a.workers<1 or a.bootstrap_reps<2: ap.error('workers >= 1 and bootstrap-reps >= 2 are required')
    if out==src or src.is_relative_to(out): ap.error('Place the input workbook outside the output directory.')
    out.mkdir(parents=True,exist_ok=True)
    digest=lambda:hashlib.sha256(src.read_bytes()).hexdigest()
    before=digest(); env=os.environ.copy()
    env.update(MICRO_INPUT=str(src),MICRO_OUTPUT=str(out),MICRO_MODE=a.mode,
        MICRO_BOOTSTRAP_REPS=str(a.bootstrap_reps),MICRO_WORKERS=str(a.workers),PYTHONUTF8='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.time(); completed=[]
    history_path=out/'run_history.json'
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    for step in a.steps:
        print(f'=== {step} ===',flush=True)
        subprocess.run([sys.executable,'-m','analysis.'+step],cwd=Path(__file__).resolve().parent,env=env,check=True)
        completed.append(step)
        history.append(dict(stage=step,mode=a.mode,input_sha256=before,bootstrap_reps=a.bootstrap_reps,completed_unix=time.time()))
        history_path.write_text(json.dumps(history,indent=2),encoding='utf-8')
    assert digest()==before,'The input workbook changed during execution.'
    (out/'run_manifest.json').write_text(json.dumps(dict(mode=a.mode,completed=completed,
        input_sha256=before,input_unchanged=True,bootstrap_reps=a.bootstrap_reps,
        workers=a.workers,elapsed_seconds=time.time()-start,python=sys.version),indent=2),encoding='utf-8')
if __name__=='__main__':main()
