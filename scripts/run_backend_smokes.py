from pathlib import Path
import os,sys,subprocess,json,time
root=Path(__file__).resolve().parents[1]
os.chdir(root)
out=root/'.integration-staging/backend_training';out.mkdir(exist_ok=True)
environment=dict(os.environ,JAX_PLATFORMS='cpu',MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root/'src'))
results=[]
for path in sorted((root/'configs').glob('*.yaml')):
    if len(sys.argv)>1 and path.stem not in sys.argv[1:]:continue
    log=out/(path.stem+'.log')
    start=time.monotonic()
    with log.open('w') as f:
        try:
            proc=subprocess.run([sys.executable,'scripts/smoke_backend_training.py',str(path)],env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=90)
            code=proc.returncode
        except subprocess.TimeoutExpired:code=124
    result=dict(config=path.name,exit_code=code,seconds=round(time.monotonic()-start,2))
    results.append(result)
    (out/'suite.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(result),flush=True)
