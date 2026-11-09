"""Private stage runner. No thesis orchestration or external service is imported."""
from pathlib import Path
from time import monotonic
import json
import numpy as np
import pandas as pd
from .runtime import (read_json, write_json, read_rows, write_rows, digest,
                      file_hash, receipt, require, utc)


def describe(root, cfg, data_root):
    from .data import describe_dataset
    if "describe" in read_json(root/"run.json")["stages"]:
        require(root,"describe");return
    files=[]
    for unit in cfg["datasets"]:
        out=root/"datasets"/unit
        describe_dataset(unit,data_root,out,demo=unit=="demo")
        files.extend([out/"manifest.json",out/"ontology.ttl",out/"ontology-validation.json"])
        manifest=read_json(out/"manifest.json")
        files.extend(out/r["path"] for r in manifest["recordings"])
    receipt(root,"describe",files)


def prepare(root,cfg):
    from .data import load_manifest,read_recording,training_segments
    from .training import TrainingScaler
    from .scenarios import source_periods,study_cases,temporal_periods,temporal_cases,materialize
    require(root,"describe")
    if "prepare" in read_json(root/"run.json")["stages"]:
        require(root,"prepare");return
    files=[];observables=[];development=[];truth=[];allperiods=[];temporal={}
    for unit in cfg["datasets"]:
        manifest=load_manifest(root/"datasets"/unit/"manifest.json")
        scaler=TrainingScaler.fit(training_segments(manifest,"train"))
        if np.any(scaler.ranges<=0): raise ValueError("Nonvarying sensor channel: "+unit)
        periods=source_periods(manifest,cfg)
        frames={}
        def frame_for(period):
            key=period["recording_id"]
            if key not in frames: frames[key]=read_recording(manifest,key,verify_hash=True)
            return frames[key].iloc[period["start"]:period["end"]].reset_index(drop=True)
        temporal_p,temporal[unit]=temporal_periods(manifest,periods,frame_for,cfg)
        cases=study_cases(periods,manifest["features"],cfg)+temporal_cases(temporal_p,len(manifest["features"]))
        periods+=temporal_p;lookup={row["period_id"]:row for row in periods}
        for period in periods:
            frame=frame_for(period)
            path=root/"periods"/unit/(period["period_id"]+".parquet")
            path.parent.mkdir(parents=True,exist_ok=True)
            # Full context stays here. Only declared sensors enter numerical model input.
            frame.loc[:,manifest["context_columns"]].to_parquet(path,index=False);files.append(path)
            clean_path=root/"evaluator"/"periods"/unit/(period["period_id"]+".npz")
            clean_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(clean_path,clean=frame.loc[:,manifest["features"]].to_numpy(float));files.append(clean_path)
        for case in cases:
            period=lookup[case["period_id"]];frame=frame_for(period)
            clean=frame.loc[:,manifest["features"]].to_numpy(float)
            raw=materialize(clean,case,scaler.ranges)
            path=root/"observations"/unit/(case["case_id"]+".npz")
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,raw=raw);files.append(path)
            obs=dict(case_id=case["case_id"],dataset=unit,period_id=period["period_id"],
              raw_path=str(path.relative_to(root)),context_path=f"periods/{unit}/{period['period_id']}.parquet",
              raw_sha256=file_hash(path),context_status=case["context_status"],panel=case["panel"])
            # No injected target, family or clean-reference array is in this input record.
            if case["role"]=="assessment": observables.append(obs);truth.append(case)
            else: development.append(dict(case=case,observable=obs))
        allperiods+=periods
    files.extend([root/"plans"/"observable.jsonl",root/"plans"/"development.jsonl",
                  root/"evaluator"/"cases.jsonl",root/"plans"/"periods.jsonl",root/"plans"/"temporal.json"])
    write_rows(files[-5],observables);write_rows(files[-4],development)
    write_rows(files[-3],truth);write_rows(files[-2],allperiods);write_json(files[-1],temporal)
    identities=[r["case_id"] for r in observables]+[r["case"]["case_id"] for r in development]
    if len(set(identities))!=len(identities): raise ValueError("Duplicate case identity")
    receipt(root,"prepare",files,planned_assessment=len(observables),planned_development=len(development),
       materialization_boundary="Fault generator runs before calibration/prediction. Truth is stored under evaluator; policy APIs receive raw observations and context only.")


def train(root,cfg,device="cpu"):
    from .data import load_manifest,training_segments
    from .training import train_bundle
    require(root,"prepare")
    if "train" in read_json(root/"run.json")["stages"]:
        require(root,"train");return
    files=[];elapsed=0.
    for unit in cfg["datasets"]:
        manifest=load_manifest(root/"datasets"/unit/"manifest.json")
        segments={split:training_segments(manifest,split) for split in ("train","validation","calibration")}
        for seed in sorted(set(cfg["seeds"]+[cfg["development_seed"]])):
            out=root/"models"/unit/f"seed-{seed}"
            started=monotonic()
            value=train_bundle(segments["train"],segments["validation"],segments["calibration"],
              feature_names=manifest["features"],config=cfg["training"],seed=seed,out=out,
              device=device,resume=(out/"last.pt").exists())
            elapsed+=monotonic()-started
            if not value["completed"]: raise RuntimeError("Training interrupted; rerun train to resume identical bindings")
            files.extend(out/name for name in ("bundle.json","best.pt","scaler.npz"))
    receipt(root,"train",files,device=device,compute_wall_seconds=elapsed)


def _metadata(root,unit,bundle):
    manifest=read_json(root/"datasets"/unit/"manifest.json")
    return dict(dataset=unit,unit=unit,context_unit=manifest.get("context_unit",unit),
      features=manifest["features"],owners=manifest["owners"],connections=manifest["connections"],
      context_columns=manifest.get("context_columns",[]),
      training_std=bundle.training_std.tolist(),training_ranges=bundle.training_ranges.tolist(),
      native_threshold=bundle.native_threshold)


def _payload(root,obs,bundle,cfg,seed,*,candidates=True):
    from .numerics import compute_evidence
    directory=root/"evidence"/obs["dataset"]/f"seed-{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(obs["case_id"]+".npz");note=path.with_suffix(".json")
    binding=dict(observation=obs["raw_sha256"],model=bundle.metadata["weights_sha256"],
                 scaler=bundle.metadata["scaler_sha256"],tto=cfg["tto"],candidates=candidates)
    if note.exists():
        info=read_json(note)
        if info["binding"]!=binding: raise ValueError("Cached evidence binding changed")
        if info["status"]!="completed": raise RuntimeError("Terminal evidence failure retained: "+str(info.get("error")))
        if file_hash(path)!=info["payload_sha256"]: raise ValueError("Numerical cache changed")
        with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files},info
    if file_hash(root/obs["raw_path"])!=obs["raw_sha256"]: raise ValueError("Observation changed")
    with np.load(root/obs["raw_path"],allow_pickle=False) as z: raw=z["raw"]
    start=monotonic()
    try:
        arrays,detail=compute_evidence(bundle,raw,tto_config=cfg["tto"],candidates=candidates)
        np.savez_compressed(path,**arrays)
        info=dict(binding=binding,status="completed",detail=detail,payload_sha256=file_hash(path),
                  path=str(path.relative_to(root)),elapsed_seconds=monotonic()-start)
        write_json(note,info)
        return arrays,info
    except Exception as exc:
        write_json(note,dict(binding=binding,status="failed",error=type(exc).__name__+": "+str(exc),
                            elapsed_seconds=monotonic()-start))
        raise


def _clean(root,unit,period_id):
    with np.load(root/"evaluator"/"periods"/unit/(period_id+".npz"),allow_pickle=False) as z:
        return z["clean"]


def _development_records(root,cfg,unit,seed,bundle):
    records=[];failures=[]
    for row in read_rows(root/"plans"/"development.jsonl"):
        case=row["case"];obs=row["observable"]
        if case["dataset"]!=unit: continue
        try:
            payload,info=_payload(root,obs,bundle,cfg,seed,candidates=case["role"]!="dictionary")
            frame=pd.read_parquet(root/obs["context_path"])
            record=dict(case=case,frame=frame,payload=payload,
                 clean=_clean(root,unit,case["period_id"]),
                 source_group=case["period_id"],native_threshold=bundle.native_threshold,
                 training_std=bundle.training_std,training_ranges=bundle.training_ranges,
                 dataset=unit,unit=unit)
            records.append(record)
            if case["role"]=="dictionary" and case["family"]=="unchanged":
                records.append(dict(record,case=dict(case,role="normal_reference")))
        except Exception as exc:
            failures.append(dict(case_id=case["case_id"],error=str(exc)))
    return records,failures


def calibrate(root,cfg,device="cpu"):
    from .training import load_bundle
    from .calibration import fit_profiles,fit_pooled_selector,attach_pooled_selector
    require(root,"train")
    if "calibrate" in read_json(root/"run.json")["stages"]:
        require(root,"calibrate");return
    files=[];records_by={};metadata_by={};profiles_by={};devseed=cfg["development_seed"]
    start=monotonic()
    for unit in cfg["datasets"]:
        bundle=load_bundle(root/"models"/unit/f"seed-{devseed}",device=device)
        records,failures=_development_records(root,cfg,unit,devseed,bundle)
        metadata=_metadata(root,unit,bundle)
        profile=fit_profiles(records,metadata=metadata,seed=devseed)
        records_by[unit]=records;metadata_by[unit]=metadata;profiles_by[unit]=profile
        write_json(root/"profiles"/unit/"development-failures.json",failures)
        files.append(root/"profiles"/unit/"development-failures.json")
    # A single pooled boosted selector is fitted across the declared units on seed23.
    selector=fit_pooled_selector([r for records in records_by.values() for r in records],
               metadata_by,profiles_by_unit=profiles_by)
    write_json(root/"profiles"/"pooled-selector.json",selector);files.append(root/"profiles"/"pooled-selector.json")
    for unit in cfg["datasets"]:
        profile=attach_pooled_selector(profiles_by[unit],records_by[unit],metadata=metadata_by[unit],selector=selector)
        path=root/"profiles"/unit/f"seed-{devseed}.json";write_json(path,profile);files.append(path)
        for seed in cfg["seeds"]:
            if seed==devseed: continue
            bundle=load_bundle(root/"models"/unit/f"seed-{seed}",device=device)
            records,failures=_development_records(root,cfg,unit,seed,bundle)
            transferred=fit_profiles(records,metadata=_metadata(root,unit,bundle),seed=seed,
                                     ranker_bundle=profile["ranker_bundle"])
            path=root/"profiles"/unit/f"seed-{seed}.json";write_json(path,transferred);files.append(path)
            path=root/"profiles"/unit/f"seed-{seed}-development-failures.json";write_json(path,failures);files.append(path)
    # Bind every numerical development payload, not only its fitted summary.
    files.extend(sorted((root/"evidence").rglob("*.json")))
    files.extend(sorted((root/"evidence").rglob("*.npz")))
    receipt(root,"calibrate",files,compute_wall_seconds=monotonic()-start,
       freeze_utc=utc(),fit_seed=devseed,prediction_labels_permitted=False)


def predict(root,cfg,device="cpu"):
    from .training import load_bundle
    from .policies import predict_all,METHODS
    require(root,"calibrate")
    if (root/"predictions"/"closed.json").exists():
        require(root,"predict");return
    planned=[];files=[];start=monotonic()
    for unit in cfg["datasets"]:
        for seed in cfg["seeds"]:
            bundle=load_bundle(root/"models"/unit/f"seed-{seed}",device=device)
            profile=read_json(root/"profiles"/unit/f"seed-{seed}.json")
            metadata=_metadata(root,unit,bundle)
            binding=digest(dict(model=bundle.metadata["weights_sha256"],profile=profile,metadata=metadata))
            for obs in read_rows(root/"plans"/"observable.jsonl"):
                if obs["dataset"]!=unit: continue
                if obs["panel"]=="temporal" and seed not in cfg["temporal_seeds"]: continue
                path=root/"predictions"/unit/f"seed-{seed}"/(obs["case_id"]+".json")
                identity=dict(case_id=obs["case_id"],dataset=unit,seed=seed,panel=obs["panel"],period_id=obs["period_id"])
                planned.append(identity)
                if path.exists():
                    old=read_json(path)
                    if old["binding"]!=binding: raise ValueError("Terminal prediction binding changed")
                else:
                    try:
                        payload,info=_payload(root,obs,bundle,cfg,seed,candidates=True)
                        frame=pd.read_parquet(root/obs["context_path"]).loc[:,metadata["context_columns"]]
                        predictions=predict_all(payload,frame,profile,metadata=metadata,context_status=obs["context_status"])
                        result=dict(identity,binding=binding,status="completed",predictions=predictions,
                          payload=info["path"],payload_sha256=info["payload_sha256"],
                          invalid_candidates=int((~payload["valid"].astype(bool)).sum()),terminal_utc=utc())
                    except Exception as exc:
                        result=dict(identity,binding=binding,status="failed",predictions={m:dict(
                          proposed=False,detected=False,concern=True,target=None,reasons=["terminal_inference_failure"])
                          for m in METHODS},error=type(exc).__name__+": "+str(exc),terminal_utc=utc())
                    write_json(path,result)
                files.append(path)
    # Terminal failures are closed alongside successes and are never retried.
    closed=dict(schema="safeprocess-prediction-closure-v1",closed_utc=utc(),planned=planned,
       files={str(p.relative_to(root)):file_hash(p) for p in files},
       calibration_receipt=digest(require(root,"calibrate")),evaluator_joined=False)
    path=root/"predictions"/"closed.json";write_json(path,closed);files.append(path)
    receipt(root,"predict",files,compute_wall_seconds=monotonic()-start,terminal_observations=len(planned))


def evaluate(root,cfg):
    from .evaluation import evaluate_closed
    require(root,"predict")
    if "evaluate" in read_json(root/"run.json")["stages"]:
        require(root,"evaluate");return
    files=evaluate_closed(root)
    receipt(root,"evaluate",files,prediction_closure_sha256=file_hash(root/"predictions"/"closed.json"))


def ablate_history(root,cfg):
    from .training import load_bundle
    from .calibration import fit_history_ladder,predict_history_ladder
    from .evaluation import outcome,summarize
    require(root,"evaluate")
    if "ablate-history" in read_json(root/"run.json")["stages"]:
        require(root,"ablate-history");return
    files=[];rows=[]
    if not cfg.get("history_ablation"):
        path=root/"ablation"/"status.json";write_json(path,dict(status="disabled"));return receipt(root,"ablate-history",[path])
    # No model.forward, compute_evidence or TTO call is allowed in this stage.
    truth={c["case_id"]:c for c in read_rows(root/"evaluator"/"cases.jsonl")}
    obs={c["case_id"]:c for c in read_rows(root/"plans"/"observable.jsonl")}
    for unit in cfg["datasets"]:
        for seed in cfg["seeds"]:
            # Bundle loading reads scaler/metadata and weights but never executes it.
            bundle=load_bundle(root/"models"/unit/f"seed-{seed}",device="cpu")
            metadata=_metadata(root,unit,bundle);records=[];missing=[]
            for record in read_rows(root/"plans"/"development.jsonl"):
                if record["case"]["dataset"]!=unit: continue
                path=root/"evidence"/unit/f"seed-{seed}"/(record["case"]["case_id"]+".npz")
                if not path.exists(): missing.append(str(path.relative_to(root)));continue
                with np.load(path,allow_pickle=False) as z: payload={k:z[k] for k in z.files}
                frame=pd.read_parquet(root/record["observable"]["context_path"])
                value=dict(case=record["case"],payload=payload,frame=frame,
                  clean=_clean(root,unit,record["case"]["period_id"]),source_group=record["case"]["period_id"],dataset=unit,unit=unit,
                  training_std=bundle.training_std,training_ranges=bundle.training_ranges,native_threshold=bundle.native_threshold)
                records.append(value)
                if record["case"]["role"]=="dictionary" and record["case"]["family"]=="unchanged":
                    records.append(dict(value,case=dict(record["case"],role="normal_reference")))
            if missing:
                path=root/"ablation"/unit/f"seed-{seed}"/"status.json";write_json(path,dict(status="unavailable",reason="incomplete_numerical_caches",missing=missing));files.append(path);continue
            profiles=read_json(root/"profiles"/unit/f"seed-{seed}.json")
            ladder=fit_history_ladder(records,metadata=metadata,profiles=profiles)
            path=root/"ablation"/unit/f"seed-{seed}"/"profiles.json";write_json(path,ladder);files.append(path)
            closed=read_json(root/"predictions"/"closed.json")
            for rel in closed["files"]:
                terminal=read_json(root/rel)
                if terminal["dataset"]!=unit or terminal["seed"]!=seed: continue
                c=truth[terminal["case_id"]];ob=obs[c["case_id"]]
                if terminal["status"]!="completed":
                    for level in ("intercept","shared","state","full"):
                        for comparison in ("risk_refitted","fixed_full_gap"):
                            rows.append(outcome(terminal,c,dict(proposed=False,concern=True),None,
                              _clean(root,unit,c["period_id"]),bundle.training_ranges,method=level,comparison=comparison))
                    continue
                with np.load(root/terminal["payload"],allow_pickle=False) as z: payload={k:z[k] for k in z.files}
                frame=pd.read_parquet(root/ob["context_path"]).loc[:,metadata["context_columns"]]
                predictions=predict_history_ladder(payload,frame,ladder,metadata=metadata,context_status=ob["context_status"])
                clean=_clean(root,unit,c["period_id"])
                for level,comparisons in predictions.items():
                    for comparison,pred in comparisons.items():
                        rows.append(outcome(terminal,c,pred,payload,clean,bundle.training_ranges,
                                            method=level,comparison=comparison))
    path=root/"ablation"/"outcomes.jsonl";write_rows(path,rows);files.append(path)
    path=root/"ablation"/"summary.csv";summarize(rows,["method","comparison","panel"]).to_csv(path,index=False);files.append(path)
    receipt(root,"ablate-history",files,scope="explanatory exposed-data analysis; both comparisons change event admission; no isolated causal feature effect",device="cpu",additional_inference=False)


def report(root,cfg):
    from .reporting import make_report
    require(root,"evaluate")
    if "report" in read_json(root/"run.json")["stages"]:
        require(root,"report");return
    receipt(root,"report",make_report(root))


def verify(root,cfg):
    from .evaluation import verify_outputs
    for stage in ("describe","prepare","train","calibrate","predict","evaluate","report"):
        require(root,stage)
    result=verify_outputs(root)
    path=root/"verification.json";write_json(path,result)
    receipt(root,"verify",[path])
    if not result["passed"]: raise RuntimeError("Run verification failed: "+str(result))
    return result
