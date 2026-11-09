"""Evaluator-only clean-reference joins and signed-error accounting."""
import numpy as np
import pandas as pd
from .runtime import read_json,read_rows,write_json,write_rows,file_hash,utc


def outcome(terminal,case,pred,payload,clean,ranges,*,method,comparison="primary"):
    target=pred.get("target",pred.get("target_index_recommended"))
    proposed=bool(pred.get("proposed",False))
    if not proposed: target=None
    true=case.get("target_index")
    native=case["family"] in ("native_process","native_attack")
    valid_estimate=bool(payload is not None and proposed and target is not None and
                        0<=int(target)<len(ranges) and np.asarray(payload["valid"],bool)[int(target)])
    evaluable=not native and payload is not None and (not proposed or valid_estimate)
    selected=None;remaining=None;observed_error=None;after_error=None
    raw=None if payload is None else np.asarray(payload["raw"])[-60:]
    clean=np.asarray(clean)[-60:]
    divisor=np.where(np.asarray(ranges)>0,ranges,1.)
    if not native and payload is not None:
        selected=0. if evaluable else None
        if proposed and valid_estimate:
            if target is None or not 0<=int(target)<len(ranges): raise ValueError("Proposal has no valid target")
            target=int(target)
            corrected=np.asarray(payload["corrected"])[target]
            observed_error=float(np.abs(raw[:,target]-clean[:,target]).mean()/divisor[target])
            after_error=float(np.abs(corrected[:,target]-clean[:,target]).mean()/divisor[target])
            selected=observed_error-after_error
        damaged=case.get("corrupted_indices",[true] if true is not None else [])
        if damaged:
            errors=[]
            for j in damaged:
                values=np.asarray(payload["corrected"])[target,:,j] if proposed and valid_estimate else raw[:,j]
                errors.append(float(np.abs(values-clean[:,j]).mean()/divisor[j]))
            remaining=float(np.mean(errors))
    correct=bool(proposed and true is not None and int(target)==int(true))
    return dict(case_id=case["case_id"],dataset=case["dataset"],seed=terminal["seed"],
      period_id=case["period_id"],panel=("unchanged" if case["family"]=="unchanged" and case["panel"]=="primary" else case["panel"]),family=case["family"],method=method,
      comparison=comparison,status=terminal["status"],detected=bool(pred.get("detected",pred.get("numeric_alarm",False))),
      proposed=proposed,target=target,true_target=true,correct_target=correct,
      correct_beneficial=bool(correct and selected is not None and selected>0),
      strict_harm=bool(proposed and selected is not None and selected<0),
      quality_evaluable=evaluable,selected_candidate_valid=valid_estimate if proposed else None,signed_benefit=selected,
      observed_selected_error=observed_error,reconstructed_selected_error=after_error,
      remaining_injected_error=remaining,concern=bool(pred.get("concern",False)),
      preview=pred.get("preview") is not None,process_flag=bool(pred.get("F",pred.get("process_flag",False))),
      context_valid=bool(pred.get("U",pred.get("context_valid",True))),
      native_category=case.get("native_category"),response_supported=case.get("response_supported",False))


def summarize(rows,by=("method","panel")):
    columns=list(by)+["N","completed","failed","events","proposals","correct_targets","beneficial",
     "harm","quality_evaluable","concern","previews","process_flags","coverage",
     "target_precision","beneficial_coverage","harm_per_quality_proposal","signed_benefit",
     "remaining_injected_error","normal_damage_mean"]
    if not rows: return pd.DataFrame(columns=columns)
    frame=pd.DataFrame(rows);result=[]
    for keys,g in frame.groupby(list(by),dropna=False,sort=True):
        if not isinstance(keys,tuple):keys=(keys,)
        n=len(g);proposals=int(g.proposed.sum());quality=int((g.proposed & g.quality_evaluable).sum())
        normal=g[(g.family=="unchanged") & g.quality_evaluable]
        result.append(dict(zip(by,keys))|dict(N=n,completed=int((g.status=="completed").sum()),
          failed=int((g.status!="completed").sum()),events=int(g.detected.sum()),proposals=proposals,
          correct_targets=int(g.correct_target.sum()),beneficial=int(g.correct_beneficial.sum()),
          harm=int(g.strict_harm.sum()),quality_evaluable=int(g.quality_evaluable.sum()),
          concern=int(g.concern.sum()),previews=int(g.preview.sum()),process_flags=int(g.process_flag.sum()),
          coverage=proposals/n,target_precision=float(g.correct_target.sum()/proposals) if proposals else None,
          beneficial_coverage=float(g.correct_beneficial.sum()/n),
          harm_per_quality_proposal=float(g.strict_harm.sum()/quality) if quality else None,
          signed_benefit=float(g.signed_benefit.mean()) if g.signed_benefit.notna().any() else None,
          remaining_injected_error=float(g.remaining_injected_error.mean()) if g.remaining_injected_error.notna().any() else None,
          normal_damage_mean=float(-normal.signed_benefit.mean()) if len(normal) else None))
    return pd.DataFrame(result,columns=columns)


def evaluate_closed(root):
    closed=read_json(root/"predictions"/"closed.json")
    if closed["evaluator_joined"]: raise ValueError("Prediction closure contains evaluator data")
    truth=list(read_rows(root/"evaluator"/"cases.jsonl"));byid={c["case_id"]:c for c in truth}
    if len(byid)!=len(truth): raise ValueError("Evaluator join is not one-to-one")
    obs={c["case_id"]:c for c in read_rows(root/"plans"/"observable.jsonl")}
    rows=[]
    for rel,sha in closed["files"].items():
        if file_hash(root/rel)!=sha: raise ValueError("Sealed prediction changed")
        terminal=read_json(root/rel);case=byid[terminal["case_id"]]
        bundle=read_json(root/"models"/case["dataset"]/f"seed-{terminal['seed']}"/"bundle.json")
        with np.load(root/"models"/case["dataset"]/f"seed-{terminal['seed']}"/"scaler.npz",allow_pickle=False) as z:
            ranges=z["ranges"] if "ranges" in z.files else z["maximum"]-z["minimum"]
        payload=None
        if terminal["status"]=="completed":
            if file_hash(root/terminal["payload"])!=terminal["payload_sha256"]: raise ValueError("Sealed payload changed")
            with np.load(root/terminal["payload"],allow_pickle=False) as z: payload={k:z[k] for k in z.files}
        with np.load(root/"evaluator"/"periods"/case["dataset"]/(case["period_id"]+".npz"),allow_pickle=False) as z: clean=z["clean"]
        for method,pred in terminal["predictions"].items():
            rows.append(outcome(terminal,case,pred,payload,clean,ranges,method=method))
    out=root/"evaluation";out.mkdir(parents=True,exist_ok=True);files=[]
    for name,groups in (("summary",["method","panel"]),("by_configuration",["dataset","method","panel"]),
                        ("by_checkpoint",["dataset","seed","method","panel"]),
                        ("by_source_period",["dataset","period_id","method","panel"]),
                        ("by_fault_family",["family","method","panel"])):
        path=out/(name+".csv");summarize(rows,groups).to_csv(path,index=False);files.append(path)
    path=out/"outcomes.jsonl";write_rows(path,rows);files.append(path)
    path=out/"join.json";write_json(path,dict(evaluated_utc=utc(),closed_utc=closed["closed_utc"],
      closure_sha256=file_hash(root/"predictions"/"closed.json"),rows=len(rows),
      observations=len(closed["planned"]),join="one case/seed to seven policies",
      weighting="pooled case matrix; source periods, channels and checkpoints are dependent",
      native_counterfactual="missing; never imputed as zero"));files.append(path)
    return files


def verify_outputs(root):
    from .policies import METHODS
    closed=read_json(root/"predictions"/"closed.json");rows=list(read_rows(root/"evaluation"/"outcomes.jsonl"))
    planned={(r["dataset"],r["seed"],r["case_id"]) for r in closed["planned"]}
    terminal=[];failures=[]
    for rel,sha in closed["files"].items():
        if file_hash(root/rel)!=sha: failures.append("prediction_hash:"+rel)
        r=read_json(root/rel);terminal.append((r["dataset"],r["seed"],r["case_id"]))
        if set(r["predictions"])!=set(METHODS): failures.append("method_inventory:"+rel)
        if r["status"] not in ("completed","failed"): failures.append("nonterminal:"+rel)
        if r["status"]=="completed" and file_hash(root/r["payload"])!=r["payload_sha256"]: failures.append("payload_hash:"+rel)
    if len(terminal)!=len(set(terminal)) or set(terminal)!=planned: failures.append("terminal_coverage")
    identities={(r["dataset"],r["seed"],r["case_id"],r["method"]) for r in rows}
    if len(identities)!=len(rows) or len(rows)!=len(planned)*len(METHODS): failures.append("one_to_one_evaluator_join")
    for row in rows:
        if row["family"] in ("native_process","native_attack") and row["signed_benefit"] is not None: failures.append("native_counterfactual_leak")
        if not row["proposed"] and row["signed_benefit"] not in (0.,None): failures.append("nonzero_abstention")
    forbidden={"target_index","true_target","clean","family","amplitude","correct_beneficial"}
    if any(forbidden.intersection(r) for r in read_rows(root/"plans"/"observable.jsonl")): failures.append("evaluator_field_in_observable_plan")
    join=read_json(root/"evaluation"/"join.json")
    if join["evaluated_utc"]<closed["closed_utc"]: failures.append("evaluation_precedes_closure")
    return dict(passed=not failures,failures=failures,planned_observations=len(planned),outcome_rows=len(rows),
      terminal_failures=sum(r["status"]!="completed" for r in rows)//len(METHODS),
      checks=["bound stage hashes","terminal coverage","seven policies","one-to-one joins","native missingness","abstention loss","observable/evaluator separation","closure chronology"])
