from pathlib import Path
import shutil
import json
from workflow.experiment import load_experiment, expand

root = Path(__file__).resolve().parents[1]
backup = sorted((root / '.integration-staging').glob('configs_before_family_*'))[-1]
for name in ['prepare_jax_campaign.py', 'launch_jax_campaign.py', 'jax_preflight.py']:
    source = root / 'scripts' / name
    if source.exists():
        (backup / 'scripts').mkdir(exist_ok=True)
        shutil.move(str(source), str(backup / 'scripts' / name))
summary = {}
for path in sorted((root / 'configs').glob('*.yaml')):
    jobs = expand(load_experiment(path))
    summary[path.name] = dict(jobs=len(jobs), algorithms=sorted({j.algo for j in jobs}),
                             envs=sorted({j.env for j in jobs}))
    if path.stem == 'minatar':
        assert len(jobs) == 50
        assert all(j.config['env_params']['sticky_action_prob'] == .1 for j in jobs)
        assert all(j.config['algo_params']['gae_lambda'] == .95 for j in jobs if j.algo == 'ppo')
        assert len({j.label.split('|')[0] for j in jobs}) == 2
print(json.dumps(summary, indent=2))
(root / '.integration-staging/family_plan_summary.json').write_text(json.dumps(summary, indent=2))
