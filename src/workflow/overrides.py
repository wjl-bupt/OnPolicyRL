"""Materialize local configuration overrides before optional revision isolation."""
import hashlib,json
from pathlib import Path
import yaml


def materialize(config, assignments, directory=None):
    source=Path(config).resolve()
    data=yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or data.get('version')!=2:
        raise ValueError('--set requires a version 2 family configuration')
    data.setdefault('name',source.stem)  # retain original default experiment name
    for assignment in assignments:
        key,sep,value=assignment.partition('=')
        parts=key.split('.')
        if not sep or any(not p for p in parts):raise ValueError(f'Expected path=value: {assignment}')
        node=data
        for part in parts[:-1]:
            node=node.setdefault(part,{})
            if not isinstance(node,dict):raise ValueError(f'Cannot descend into non-mapping: {key}')
        node[parts[-1]]=yaml.safe_load(value)
    content=json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+'\n'
    target=Path(directory or '.oprl/launch_configs')
    target.mkdir(parents=True,exist_ok=True)
    path=target/(hashlib.sha256(content.encode()).hexdigest()+'.json')
    if path.exists() and path.read_text(encoding='utf-8')!=content:raise RuntimeError('Snapshot content mismatch')
    if not path.exists():path.write_text(content,encoding='utf-8')
    return str(path.resolve())
