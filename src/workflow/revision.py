"""Run committed source with independently snapshotted local configuration.

Never checks out, cleans, resets, or commits the user's working tree. Git
objects are fetched only when missing. Source archives exclude configs/.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid


def _git(repo, *args, check=True):
    return subprocess.run(['git', '-C', str(repo), *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def resolve_commit(repo, revision, remote='origin'):
    if not re.fullmatch(r'[0-9a-fA-F]{7,40}', revision):
        raise ValueError('Use a commit ID (7..40 hex characters), not a mutable branch/tag')
    result = _git(repo, 'rev-parse', '--verify', revision + '^{commit}', check=False)
    if result.returncode:
        # The named remote must already exist in this repository.
        _git(repo, 'remote', 'get-url', remote)
        args = ['fetch', '--no-tags', remote]
        if len(revision) == 40:
            args.append(revision)
        _git(repo, *args)
        result = _git(repo, 'rev-parse', '--verify', revision + '^{commit}')
    return result.stdout.decode().strip()


def _hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()
            and p.name != '.source-manifest.json'
            and '__pycache__' not in p.parts}


def materialize(repo, sha, cache):
    cache = Path(cache).resolve();cache.mkdir(parents=True, exist_ok=True)
    target = cache / sha
    if target.exists():
        manifest = json.loads((target / '.source-manifest.json').read_text())
        if manifest['commit'] != sha or manifest['files'] != _hashes(target):
            raise ValueError(f'Source cache was modified: {target}; use a fresh --code-cache')
        return target
    archive = _git(repo, 'archive', '--format=tar', sha).stdout
    staging = Path(tempfile.mkdtemp(prefix=sha + '-', dir=cache))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar:
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe archive path')
                if relative.parts and relative.parts[0] in ('configs', 'runs', '.git'):
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f'Source archive links are not supported: {member.name}')
                dest = staging / relative;dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
                dest.chmod(member.mode & 0o777)
        if not (staging / 'src/workflow/runner.py').is_file():
            raise ValueError('Selected commit does not contain src/workflow/runner.py')
        manifest = {'commit': sha, 'files': _hashes(staging)}
        (staging / '.source-manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            staging.rename(target)
        except OSError:
            if not target.exists():raise
            return materialize(repo, sha, cache)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return target


_BOOTSTRAP = r'''
import json, os, sys, time, traceback
from pathlib import Path
manifest_path=Path(sys.argv[1])
manifest=json.loads(manifest_path.read_text())
def save():
    tmp=manifest_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(manifest,indent=2))
    tmp.replace(manifest_path)
manifest.update(status='running',pid=os.getpid(),started=time.time());save()
try:
    import workflow.runner as runner
    path=Path(runner.__file__).resolve()
    path.relative_to(Path(manifest['source_dir']).resolve())
    manifest['executed_runner']=str(path);save()
    results=runner.run(manifest['runtime_config'],device=manifest['device'])
    code=int(any(r['status']=='failed' for r in results))
    manifest.update(status='failed' if code else 'done',results=results)
except BaseException:
    traceback.print_exc();manifest.update(status='failed',error=traceback.format_exc());code=1
manifest.update(finished=time.time(),exit_code=code);save()
raise SystemExit(code)
'''


def prepare_revision(config, revision, *, repo=None, remote='origin', code_cache=None,
                     run_root=None, device=None):
    repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
    config = Path(config).resolve()
    raw = config.read_bytes()
    # Read the snapshotted bytes rather than reopening a mutable source later.
    suffix = config.suffix.lower()
    if suffix == '.json':data = json.loads(raw)
    elif suffix in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(raw)
    elif suffix == '.toml':
        import tomllib
        data = tomllib.loads(raw.decode())
    else:raise ValueError('Configuration must be YAML, JSON or TOML')
    if not isinstance(data, dict):raise ValueError('Configuration must be a mapping')
    sha = resolve_commit(repo, revision, remote)
    source = materialize(repo, sha, code_cache or repo / '.oprl/code')
    digest = hashlib.sha256(raw).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    output = Path(run_root or repo / 'runs/by_commit').resolve() / sha[:12] / digest[:12] / stamp
    output.mkdir(parents=True)
    snapshot = output / ('config.snapshot' + suffix);snapshot.write_bytes(raw)
    # The caller's run_dir is recorded but relocated to avoid cross-version
    # checkpoint reuse, regardless of resume=true in the editable config.
    configured_run_dir = data.get('run_dir')
    data['run_dir'] = str(output / 'results')
    runtime = output / 'config.runtime.json';runtime.write_text(json.dumps(data, indent=2))
    from importlib import metadata
    manifest = dict(commit=sha, config_sha256=digest, source_dir=str(source),
        local_config=str(config), config_snapshot=str(snapshot), runtime_config=str(runtime),
        requested_run_dir=configured_run_dir, run_dir=data['run_dir'], device=device,
        python=sys.executable, status='prepared', source_manifest=str(source / '.source-manifest.json'),
        versions={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata['Name']})
    path = output / 'manifest.json';path.write_text(json.dumps(manifest, indent=2))
    return path


def run_revision(config, revision, *, prepare_only=False, detach=False, **kwargs):
    path = prepare_revision(config, revision, **kwargs)
    manifest = json.loads(path.read_text())
    print(f"Code commit: {manifest['commit']}\nConfig SHA256: {manifest['config_sha256']}\nManifest: {path}", flush=True)
    if prepare_only:return 0
    source = Path(manifest['source_dir'])
    env = os.environ.copy()
    # Put only the selected commit on PYTHONPATH; do not inherit the local source.
    env.update(PYTHONPATH=str(source/'src'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OPRL_SOURCE_COMMIT=manifest['commit'], OPRL_CONFIG_SHA256=manifest['config_sha256'])
    with (path.parent/'launch.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen([sys.executable,'-c',_BOOTSTRAP,str(path)],env=env,
            cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=detach)
    print(f'PID: {proc.pid}; log: {path.parent / "launch.log"}', flush=True)
    return 0 if detach else proc.wait()
