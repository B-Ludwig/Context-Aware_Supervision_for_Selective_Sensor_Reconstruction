"""Acquisition must complete before a run can train or freeze input bindings."""
from pathlib import Path
import json

from safeprocess.cli import main
from safeprocess.config import resolve_config
from safeprocess.runtime import open_run


def test_fetch_only_reports_missing_provider_without_run_or_model(tmp_path,monkeypatch):
    from safeprocess import acquisition
    calls=[]
    def missing(root,datasets,**kwargs):
        calls.append((root,datasets))
        return dict(schema='test',complete=False,providers={'swat':dict(status='manual_required',message='Supply the authorized recordings')},files=[])
    monkeypatch.setattr(acquisition,'acquire_datasets',missing)
    assert main(['fetch-data','--data-root',str(tmp_path),'--datasets','swat'])==2
    assert calls[0][1]==['swat']
    assert json.loads((tmp_path/'acquisition.json').read_text())['complete'] is False
    assert not (tmp_path/'run.json').exists()


def test_run_does_not_freeze_or_train_with_unavailable_recordings(tmp_path,monkeypatch):
    from safeprocess import acquisition,workflow
    monkeypatch.setattr(acquisition,'acquire_datasets',lambda *args,**kwargs:dict(complete=False,providers={'hai':dict(status='manual_required')},files=[]))
    def forbidden(*args,**kwargs):raise AssertionError('Training must not start')
    monkeypatch.setattr(workflow,'train',forbidden)
    assert main(['run','--config','paper','--datasets','hai_ds1','--data-root',str(tmp_path/'data'),'--work-dir',str(tmp_path/'work')])==2
    assert not (tmp_path/'work/run.json').exists()


def test_frozen_resume_never_refetches_sources(tmp_path,monkeypatch):
    from safeprocess import acquisition,workflow
    work=tmp_path/'work'
    open_run(work,resolve_config('paper',datasets=['swat'],seeds=[23]))
    def forbidden(*args,**kwargs):raise AssertionError('Frozen run must not download sources')
    monkeypatch.setattr(acquisition,'acquire_datasets',forbidden)
    stages=[]
    for name in ('describe','prepare','train','calibrate','predict','evaluate','ablate_history','report','verify'):
        monkeypatch.setattr(workflow,name,lambda *args,_name=name,**kwargs:stages.append(_name))
    assert main(['run','--work-dir',str(work)])==0
    assert len(stages)==9


def test_preflight_remains_read_only_with_respect_to_acquisition(tmp_path,monkeypatch):
    from safeprocess import acquisition,data
    monkeypatch.setattr(data,'preflight_dataset',lambda *args:dict(ok=True))
    def forbidden(*args,**kwargs):raise AssertionError('Preflight must not acquire inputs')
    monkeypatch.setattr(acquisition,'acquire_datasets',forbidden)
    assert main(['describe','--preflight','--config','paper','--datasets','swat','--data-root',str(tmp_path/'data'),'--work-dir',str(tmp_path/'check')])==0
    assert (tmp_path/'check/preflight.json').exists()
    assert not (tmp_path/'data').exists()
