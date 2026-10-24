import json
from pathlib import Path
import subprocess
import pytest
from workflow.revision import prepare_revision, run_revision, resolve_commit


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT).decode().strip()


@pytest.fixture
def repo(tmp_path):
    r=tmp_path/'repo';r.mkdir();git(r,'init','-q')
    git(r,'config','user.email','test@example.invalid');git(r,'config','user.name','Test')
    (r/'src/workflow').mkdir(parents=True)
    (r/'src/workflow/runner.py').write_text('''import json
from pathlib import Path
VERSION=1
def run(config,device=None):
    data=json.loads(Path(config).read_text())
    out=Path(data['run_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'observed.json').write_text(json.dumps({'version':VERSION,'value':data['value']}))
    return [{'status':'done'}]
''')
    (r/'configs').mkdir();(r/'configs/test.json').write_text('{"value":1}')
    git(r,'add','.');git(r,'commit','-qm','first')
    return r


def test_committed_code_and_uncommitted_config_are_independent(repo,tmp_path):
    sha=git(repo,'rev-parse','HEAD')
    runner=repo/'src/workflow/runner.py'
    runner.write_text(runner.read_text().replace('VERSION=1','VERSION=999'))
    cfg=repo/'configs/test.json';cfg.write_text('{"value":42}')
    before=git(repo,'diff')
    out=tmp_path/'runs'
    assert run_revision(cfg,sha,repo=repo,run_root=out)==0
    m=json.loads(next(out.rglob('manifest.json')).read_text())
    assert m['status']=='done' and m['commit']==sha
    assert json.loads((Path(m['run_dir'])/'observed.json').read_text())=={'version':1,'value':42}
    assert not (Path(m['source_dir'])/'configs').exists()
    assert git(repo,'diff')==before
    assert git(repo,'rev-parse','HEAD')==sha
    cfg.write_text('{"value":43}')
    m2=json.loads(prepare_revision(cfg,sha,repo=repo,run_root=out).read_text())
    assert m2['source_dir']==m['source_dir'] and m2['config_sha256']!=m['config_sha256']
    assert Path(m['config_snapshot']).read_text()=='{"value":42}'


def test_modified_cached_source_rejected(repo,tmp_path):
    sha=git(repo,'rev-parse','HEAD')
    m=json.loads(prepare_revision(repo/'configs/test.json',sha,repo=repo,run_root=tmp_path/'runs').read_text())
    (Path(m['source_dir'])/'src/workflow/runner.py').write_text('changed')
    with pytest.raises(ValueError,match='modified'):
        prepare_revision(repo/'configs/test.json',sha,repo=repo)


def test_missing_commit_fetched_without_checkout(repo,tmp_path):
    client=tmp_path/'client'
    subprocess.run(['git','clone','-q',str(repo),str(client)],check=True)
    original=git(client,'rev-parse','HEAD')
    (repo/'new.py').write_text('VALUE=2')
    git(repo,'add','new.py');git(repo,'commit','-qm','second');sha=git(repo,'rev-parse','HEAD')
    assert resolve_commit(client,sha)==sha
    assert git(client,'rev-parse','HEAD')==original and not (client/'new.py').exists()


@pytest.mark.parametrize('revision',['main','HEAD','--upload-pack=evil','a'*41])
def test_only_commit_ids(repo,revision):
    with pytest.raises(ValueError):resolve_commit(repo,revision)
