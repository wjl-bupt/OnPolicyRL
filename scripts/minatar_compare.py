"""Persistent bounded GPU queue for the native PPO/GA2E MinAtar comparison."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def worker(job_file):
    from workflow.experiment import Job
    from workflow.runner import run_job
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=True
    data=json.loads(Path(job_file).read_text());data['run_dir']=Path(data['run_dir'])
    result=run_job(Job(**data),device='cuda:0')
    atomic(Path(job_file).with_suffix('.result.json'),result)
    return 0 if result['status'] in ('done','skipped') else 1

def supervise(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    jobs=manifest['jobs'];active={};pending=list(range(len(jobs)));completed=[]
    state={'pid':os.getpid(),'started':time.time(),'jobs':[{**j,'status':'queued'} for j in jobs]}
    while pending or active:
        for gpu in manifest['gpus']:
            if gpu in active or not pending:continue
            k=pending.pop(0);j=jobs[k]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root/'source/src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            log=open(root/j['console_log'],'ab',buffering=0)
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker',str(root/j['job_file'])],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(proc,k,log)
            state['jobs'][k].update(status='running',gpu=gpu,pid=proc.pid,started=time.time())
            atomic(root/'status.json',state)
        for gpu,(proc,k,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();state['jobs'][k].update(status='done' if code==0 else 'failed',exit_code=code,finished=time.time())
            del active[gpu];completed.append(k)
        state.update(updated=time.time(),running=len(active),queued=len(pending),finished=len(completed))
        atomic(root/'status.json',state)
        if active:time.sleep(15)
    # Existing native report layer aggregates the five seeds per environment.
    from workflow.experiment import load_experiment
    from workflow.report import report
    for cfg in manifest['configs']:
        try:report(load_experiment(cfg))
        except Exception as error:state.setdefault('report_errors',[]).append(str(error))
    state['complete']=True;atomic(root/'status.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','supervise']);parser.add_argument('path');args=parser.parse_args()
    if args.mode=='worker':raise SystemExit(worker(args.path))
    supervise(args.path)
