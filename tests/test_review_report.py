import json
from pathlib import Path
import pytest
from workflow.review_report import build_report,read_curve,iqm


def run(root,name,seed,values=(1.,3.,5.),times=(1.,2.,3.),done=True):
    p=root/f'{name}-{seed}';p.mkdir()
    with (p/'metrics.jsonl').open('w') as f:
        for step,v,t in zip((10,20,30),values,times):
            f.write(json.dumps(dict(step=step,eval=dict(ep_rew_mean=v,n_episodes=2,requested_episodes=2,session_wall_seconds=t,session_steps=step)))+'\n')
            f.write(json.dumps(dict(step=step,train=dict(kl=.01)))+'\n')
    if done:(p/'DONE').touch()
    return dict(path=str(p),algorithm=name,environment='test',seed=seed,budget_steps=30)


def test_report_preserves_eval_rows_and_counts_missing(tmp_path):
    a=run(tmp_path,'ppo',0);b=run(tmp_path,'ga2e',0,values=(2,4,6))
    missing=dict(b,seed=1,path=str(tmp_path/'missing'))
    result=build_report(dict(runs=[a,b,missing],comparison_reference='ppo'),tmp_path/'out')
    assert len(read_curve(Path(a['path'])/'metrics.jsonl'))==3
    g=next(x for x in result['summary'] if x['algorithm']=='ga2e')
    assert g['expected_seeds']==2 and g['complete_seeds']==1
    assert g['mean_ci95'] is None
    assert not result['win_tie_loss'] and not result['aggregate']


def test_common_time_and_explicit_normalization(tmp_path):
    rows=[run(tmp_path,'ppo',0),run(tmp_path,'ga2e',0,values=(2,4,6),times=(2,4,6))]
    result=build_report(dict(runs=rows,normalization={'test':dict(low=0,high=10,source='test fixture')},comparison_reference='ppo'),tmp_path/'out')
    assert result['aggregate']['ga2e']['iqm']==pytest.approx(.6)
    assert all(r['domain']==[2.,3.] for r in result['equal_time'])
    assert next(r for r in result['equal_time'] if r['algorithm']=='ga2e')['score_at_common_end']==pytest.approx(3.)
    assert result['win_tie_loss']['ga2e']['win']==1


def test_resume_time_and_unfinished_are_not_extrapolated(tmp_path):
    rows=[run(tmp_path,'ppo',0,times=(1,2,.5),done=False)]
    result=build_report(dict(runs=rows),tmp_path/'out')
    assert result['runs'][0]['status']=='unfinished'
    assert result['runs'][0]['time_auc'] is None
    assert result['summary'][0]['complete_seeds']==0
    assert not result['equal_time']


def test_duplicate_seed_and_implicit_normalization_rejected(tmp_path):
    row=run(tmp_path,'ppo',0)
    with pytest.raises(ValueError,match='Duplicate'):build_report(dict(runs=[row,row]),tmp_path/'out')
    with pytest.raises(ValueError,match='Normalization'):build_report(dict(runs=[row],normalization={'test':dict(low=0,high=1)}),tmp_path/'out')
    assert iqm([0,1,2,3])==1.5
