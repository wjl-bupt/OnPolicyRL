"""Native experiment CLI, optionally pinning code independently of configuration."""
import argparse


def main():
    p=argparse.ArgumentParser(description='OnPolicyRL native experiment runner')
    p.add_argument('config',nargs='?')
    p.add_argument('--list',action='store_true')
    p.add_argument('--device')
    p.add_argument('--plan', action='store_true', help='Validate and print resolved jobs without training')
    p.add_argument('--commit',help='Code commit ID; local config need not be committed')
    p.add_argument('--repo',help='Git repository to resolve/fetch the commit from')
    p.add_argument('--remote',default='origin')
    p.add_argument('--code-cache')
    p.add_argument('--run-root')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--detach',action='store_true')
    a=p.parse_args()
    if a.list:
        from workflow.registry import list_algorithms
        print(' '.join(list_algorithms()));return
    if not a.config:p.error('experiment config required')
    if a.plan:
        if a.commit:
            p.error('--plan inspects current code; use --commit --prepare-only to prepare pinned code')
        import json
        from workflow.experiment import load_experiment, expand
        jobs = expand(load_experiment(a.config))
        print(json.dumps({'jobs': len(jobs), 'plan': [dict(algo=j.algo, env=j.env,
            seed=j.seed, identity=j.identity, run_dir=str(j.run_dir), config=j.config)
            for j in jobs]}, indent=2))
        return
    if a.commit:
        from workflow.revision import run_revision
        raise SystemExit(run_revision(a.config,a.commit,repo=a.repo,remote=a.remote,
            code_cache=a.code_cache,run_root=a.run_root,device=a.device,
            prepare_only=a.prepare_only,detach=a.detach))
    if a.prepare_only or a.detach or a.repo or a.code_cache or a.run_root:
        p.error('--prepare-only/--detach/--repo/--code-cache/--run-root require --commit')
    from workflow.runner import run
    results=run(a.config,device=a.device)
    if any(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
