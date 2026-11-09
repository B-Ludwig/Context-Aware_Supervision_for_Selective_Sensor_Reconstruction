"""Independent contracts for source isolation, observation identity and closure."""
from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from safeprocess.config import resolve_config
from safeprocess.runtime import (open_run, read_json, write_json, write_rows,
                                  receipt, require, file_hash)
from safeprocess.scenarios import (source_periods, study_cases, materialize,
                                   case_for, spaced)
from safeprocess.policies import predict_all


def test_fresh_recipe_and_primary_inventory():
    cfg=resolve_config("paper")
    assert len(cfg["datasets"])==12
    assert cfg["seeds"]==[23,42,101,2027,31415]
    assert cfg["training"]["max_epochs"]==3000
    channels=[11,16,8,8,19,32,16,27,24,46,13,11]
    assert sum(cfg["assessment_counts"][unit]*d*15*5
               for unit,d in zip(cfg["datasets"],channels))==193350
    assert cfg["training"]["weight_decay"]==0
    assert cfg["training"]["post_mlp_dropout"]==0


def test_source_role_isolation_and_case_identity():
    cfg=resolve_config("demo")
    manifest=dict(dataset="demo",recordings=[dict(id="normal",normal=True,
        sha256="a",segments=[dict(start=0,end=6000,split="train"),
        dict(start=6059,end=9000,split="validation"),
        dict(start=9059,end=12000,split="calibration"),
        dict(start=12059,end=15000,split="test_normal")])])
    periods=source_periods(manifest,cfg)
    for a in periods:
        for b in periods:
            if a["recording_id"]==b["recording_id"] and a["role"]!=b["role"]:
                assert a["end"]<=b["start"] or b["end"]<=a["start"]
    cases=study_cases(periods,["a","b","c","d"],cfg)
    assert len(set(c["case_id"] for c in cases))==len(cases)
    fit={c["period_id"] for c in cases if c["role"]=="decision_fit"}
    risk={c["period_id"] for c in cases if c["role"]=="risk_calibration"}
    assert fit.isdisjoint(risk)
    assert any(c["panel"]=="context_violation" for c in cases)
    assert len(spaced(list(range(100)),24))==24


@pytest.mark.parametrize("kind",["drift","bias","gain","stuck","noisy_stuck","quantized_stuck"])
def test_corruptions_are_deterministic_and_target_local(kind):
    raw=np.arange(900).reshape(300,3)/100.
    case=dict(case_id="fixture",family=kind,target_index=1,amplitude=.1)
    before=raw.copy()
    left=materialize(raw,case,np.ones(3))
    right=materialize(raw,case,np.ones(3))
    assert np.array_equal(raw,before)
    assert np.array_equal(left,right)
    assert np.array_equal(left[:60],raw[:60])
    assert np.array_equal(left[:,[0,2]],raw[:,[0,2]])


def test_prediction_api_has_no_evaluator_arguments():
    names=set(inspect.signature(predict_all).parameters)
    assert names=={"payload","frame","profiles","metadata","context_status"}
    assert not {"target","target_index","clean","family","label"}&names


def test_run_bindings_and_immutable_stage_receipts(tmp_path):
    root,run=open_run(tmp_path/"run",resolve_config("demo"))
    assert run["run_id"].startswith("fresh-")
    assert open_run(root,resolve_config("demo"))[1]["run_id"]==run["run_id"]
    with pytest.raises(ValueError,match="configuration differs"):
        open_run(root,resolve_config("demo",seeds=[42]))
    path=root/"fixture.json"
    write_json(path,{"test":1})
    receipt(root,"fixture",[path])
    write_json(path,{"test":2})
    with pytest.raises(ValueError,match="Bound file changed"):
        require(root,"fixture")
    with pytest.raises(ValueError,match="Completed stage changed"):
        receipt(root,"fixture",[path])


def test_terminal_prediction_failure_is_never_retried(tmp_path,monkeypatch):
    from safeprocess import workflow
    from safeprocess import training
    cfg=resolve_config("demo")
    root,_=open_run(tmp_path/"run",cfg)
    for stage in ("describe","prepare","train","calibrate"):
        receipt(root,stage,[])
    write_json(root/"datasets/demo/manifest.json",dict(features=["a","b"],owners=["a","b"],
         connections=[["a","b"]],context_unit="hai_demo"))
    write_json(root/"profiles/demo/seed-23.json",{})
    write_rows(root/"plans/observable.jsonl",[dict(case_id="terminal",dataset="demo",
         period_id="period",panel="primary")])
    fake=SimpleNamespace(metadata={"weights_sha256":"weights"},training_std=np.ones(2),
         training_ranges=np.ones(2),native_threshold=1.)
    monkeypatch.setattr(training,"load_bundle",lambda *args,**kwargs:fake)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        raise RuntimeError("retained failure")
    monkeypatch.setattr(workflow,"_payload",fail)
    workflow.predict(root,cfg)
    terminal=root/"predictions/demo/seed-23/terminal.json"
    assert read_json(terminal)["status"]=="failed"
    first=file_hash(terminal)
    workflow.predict(root,cfg)
    assert len(calls)==1
    assert file_hash(terminal)==first
    assert read_json(root/"predictions/closed.json")["evaluator_joined"] is False


def _quality_fixture():
    terminal=dict(seed=23,status="completed")
    case=dict(case_id="quality",dataset="demo",period_id="period",panel="primary",
              family="drift",target_index=0)
    clean=np.zeros((300,2))
    raw=clean.copy();raw[-60:,0]=2.
    corrected=np.repeat(raw[-60:][None],2,0)
    corrected[0,:,0]=.5
    corrected[1,:,1]=1.
    payload=dict(raw=raw,corrected=corrected,valid=np.ones(2,bool))
    return terminal,case,payload,clean,np.array([2.,4.])


def test_selected_signed_error_and_remaining_wrong_target():
    from safeprocess.evaluation import outcome
    terminal,case,payload,clean,ranges=_quality_fixture()
    correct=outcome(terminal,case,dict(proposed=True,target=0),payload,clean,ranges,method="context")
    assert correct["signed_benefit"]==.75
    assert correct["remaining_injected_error"]==.25
    assert correct["correct_beneficial"] and not correct["strict_harm"]
    wrong=outcome(terminal,case,dict(proposed=True,target=1),payload,clean,ranges,method="context")
    assert wrong["signed_benefit"]==-.25
    assert wrong["remaining_injected_error"]==1.
    assert wrong["strict_harm"] and not wrong["correct_beneficial"]
    abstain=outcome(terminal,case,dict(proposed=False),payload,clean,ranges,method="context")
    assert abstain["signed_benefit"]==0.
    assert abstain["remaining_injected_error"]==1.


def test_native_counterfactual_and_failure_quality_remain_missing():
    from safeprocess.evaluation import outcome
    terminal,case,payload,clean,ranges=_quality_fixture()
    native=outcome(terminal,dict(case,family="native_process",target_index=None),
         dict(proposed=True,target=0),payload,clean,ranges,method="context")
    assert not native["quality_evaluable"]
    assert native["signed_benefit"] is None
    assert native["remaining_injected_error"] is None
    failed=outcome(dict(terminal,status="failed"),case,dict(proposed=False),None,
                   clean,ranges,method="context")
    assert not failed["quality_evaluable"]
    assert failed["signed_benefit"] is None


def test_common_mode_is_selected_channel_benefit_not_complete_recovery():
    from safeprocess.evaluation import outcome
    terminal,case,payload,clean,ranges=_quality_fixture()
    payload["raw"][-60:,1]=4.
    payload["corrected"][:, :,1]=4.
    common=outcome(terminal,dict(case,family="common_mode",target_index=None,
       corrupted_indices=[0,1]),dict(proposed=True,target=0),payload,clean,ranges,method="context")
    assert common["signed_benefit"]==.75
    assert common["remaining_injected_error"]==.625
    assert not common["correct_beneficial"]


def test_unchanged_controls_have_separate_primary_denominators():
    from safeprocess.evaluation import outcome,summarize
    terminal,case,payload,clean,ranges=_quality_fixture()
    unchanged=outcome(terminal,dict(case,family="unchanged",target_index=None),
         dict(proposed=False),payload,clean,ranges,method="context")
    fault=outcome(terminal,case,dict(proposed=False),payload,clean,ranges,method="context")
    assert unchanged["panel"]=="unchanged"
    table=summarize([unchanged,fault])
    assert table.loc[table.panel=="primary","N"].iloc[0]==1
    temporal=outcome(terminal,dict(case,family="unchanged",panel="temporal",target_index=None),
         dict(proposed=False),payload,clean,ranges,method="context")
    assert temporal["panel"]=="temporal"


def test_downstream_resume_rechecks_upstream_bindings(tmp_path):
    root,_=open_run(tmp_path/"run",resolve_config("demo"))
    source=root/"source.json"
    write_json(source,{"bound":1})
    receipt(root,"describe",[source])
    for stage in ("prepare","train","calibrate"):
        receipt(root,stage,[])
    require(root,"calibrate")
    write_json(source,{"bound":2})
    with pytest.raises(ValueError,match="Bound file changed"):
        require(root,"calibrate")


def test_assessment_policy_receives_context_without_clean_sensor_columns(tmp_path,monkeypatch):
    import pandas as pd
    from safeprocess import workflow,training,policies
    cfg=resolve_config("demo")
    root,_=open_run(tmp_path/"run",cfg)
    for stage in ("describe","prepare","train","calibrate"):
        receipt(root,stage,[])
    context_columns=["unit.state_idle.active","unit.state_work.active"]
    write_json(root/"datasets/demo/manifest.json",dict(features=["a","b"],owners=["a","b"],
         connections=[["a","b"]],context_unit="hai_demo",context_columns=context_columns))
    write_json(root/"profiles/demo/seed-23.json",{})
    frame=pd.DataFrame({"a":np.zeros(300),"b":np.ones(300),
                        context_columns[0]:np.ones(300),context_columns[1]:np.zeros(300)})
    frame.to_parquet(root/"context.parquet")
    write_rows(root/"plans/observable.jsonl",[dict(case_id="observable",dataset="demo",
         period_id="period",panel="primary",context_path="context.parquet",context_status="available")])
    fake=SimpleNamespace(metadata={"weights_sha256":"weights"},training_std=np.ones(2),
         training_ranges=np.ones(2),native_threshold=1.,feature_names=["a","b"])
    monkeypatch.setattr(training,"load_bundle",lambda *args,**kwargs:fake)
    monkeypatch.setattr(workflow,"_payload",lambda *args,**kwargs:(dict(valid=np.ones(2,bool)),
          dict(path="fake.npz",payload_sha256="fake")))
    received=[]
    def capture(payload,frame,profiles,**kwargs):
        received.append(frame.columns.tolist())
        return {method:dict(proposed=False,concern=False,detected=False,target=None)
                for method in policies.METHODS}
    monkeypatch.setattr(policies,"predict_all",capture)
    workflow.predict(root,cfg)
    assert received==[context_columns]
    assert read_json(root/"predictions/demo/seed-23/observable.json")["status"]=="completed"


def test_selection_normal_fit_is_disjoint_and_contains_no_faults():
    cfg=resolve_config("demo")
    manifest=dict(dataset="demo",recordings=[dict(id="normal",normal=True,sha256="a",
        segments=[dict(start=0,end=9000,split="train"),
        dict(start=9059,end=12000,split="validation"),
        dict(start=12059,end=15000,split="calibration"),
        dict(start=15059,end=18000,split="test_normal")])])
    periods=source_periods(manifest,cfg)
    extra=[p for p in periods if p["role"]=="selection_normal_fit"]
    assert len(extra)==cfg["roles"]["selection_normal_fit"]
    for p in extra:
        for other in periods:
            if other["role"]!="selection_normal_fit" and p["recording_id"]==other["recording_id"]:
                assert p["end"]<=other["start"] or other["end"]<=p["start"]
    cases=study_cases(periods,["a","b","c","d"],cfg)
    extra_cases=[c for c in cases if c["role"]=="selection_normal_fit"]
    assert len(extra_cases)==len(extra)
    assert all(c["family"]=="unchanged" and c["target_index"] is None for c in extra_cases)


def test_failed_assessment_remains_in_every_history_level_and_comparison(tmp_path,monkeypatch):
    from safeprocess import workflow,training,calibration,context
    cfg=resolve_config("demo")
    root,_=open_run(tmp_path/"run",cfg)
    for stage in ("describe","prepare","train","calibrate","predict","evaluate"):
        receipt(root,stage,[])
    write_json(root/"datasets/demo/manifest.json",dict(features=["a","b"],owners=["a","b"],
         connections=[["a","b"]],context_unit="hai_demo",context_columns=["state"]))
    write_json(root/"profiles/demo/seed-23.json",{})
    terminal=dict(case_id="failed",dataset="demo",seed=23,period_id="period",status="failed")
    write_json(root/"predictions/demo/seed-23/failed.json",terminal)
    write_json(root/"predictions/closed.json",dict(files={"predictions/demo/seed-23/failed.json":"irrelevant"}))
    case=dict(case_id="failed",dataset="demo",period_id="period",panel="primary",family="drift",target_index=0)
    write_rows(root/"evaluator/cases.jsonl",[case])
    write_rows(root/"plans/observable.jsonl",[dict(case_id="failed")])
    write_rows(root/"plans/development.jsonl",[])
    fake=SimpleNamespace(metadata={},training_std=np.ones(2),training_ranges=np.ones(2),native_threshold=1.)
    monkeypatch.setattr(training,"load_bundle",lambda *args,**kwargs:fake)
    monkeypatch.setattr(workflow,"_clean",lambda *args:np.zeros((300,2)))
    monkeypatch.setattr(calibration,"fit_history_ladder",lambda *args,**kwargs:dict(
          variants={level:{} for level in context.HISTORY_LEVELS}))
    workflow.ablate_history(root,cfg)
    from safeprocess.runtime import read_rows
    rows=list(read_rows(root/"ablation/outcomes.jsonl"))
    assert len(rows)==8
    assert {r["method"] for r in rows}==set(context.HISTORY_LEVELS)
    assert {r["comparison"] for r in rows}=={"risk_refitted","fixed_full_gap"}
    assert all(r["status"]=="failed" and r["signed_benefit"] is None for r in rows)


def test_invalid_proposed_candidate_has_missing_quality_and_keeps_original_error():
    from safeprocess.evaluation import outcome
    terminal,case,payload,clean,ranges=_quality_fixture()
    payload["valid"][0]=False
    invalid=outcome(terminal,case,dict(proposed=True,target=0),payload,clean,ranges,method="residual_dominance")
    assert invalid["proposed"]
    assert not invalid["quality_evaluable"]
    assert invalid["signed_benefit"] is None
    assert invalid["observed_selected_error"] is None
    assert invalid["reconstructed_selected_error"] is None
    assert invalid["remaining_injected_error"]==1.
    assert not invalid["correct_beneficial"] and not invalid["strict_harm"]
    # An invalid candidate does not erase the zero policy change of abstention.
    abstain=outcome(terminal,case,dict(proposed=False),payload,clean,ranges,method="context")
    assert abstain["signed_benefit"]==0.
    assert abstain["remaining_injected_error"]==1.


def test_resume_rejects_changed_implementation(tmp_path,monkeypatch):
    from safeprocess import runtime
    root,_=open_run(tmp_path/"run",resolve_config("demo"))
    monkeypatch.setattr(runtime,"implementation_identity",lambda:"different-source-bindings")
    with pytest.raises(ValueError,match="implementation differs"):
        open_run(root,resolve_config("demo"))
