"""Deterministic source planning and versioned paper corruptions.

This module produces observations and separate evaluator records. No truth field
is consumed by a policy. Times are row coordinates at the recordings' sample rate.
"""
from collections import Counter
import hashlib
import numpy as np
from .runtime import digest

FIT = [("drift",a) for a in (-20.,-5.,-1.,1.,5.,20.)]+[("stuck",0.)]
RISK = [("drift",a) for a in (-50.,-10.,-2.,2.,10.,50.)]+[("stuck",0.)]
ASSESS = [("drift",a) for a in (-50.,-20.,-10.,-5.,-2.,-1.,1.,2.,5.,10.,20.,50.)]+[("stuck",0.),("noisy_stuck",.1),("quantized_stuck",.1)]
TEMPORAL = [("drift",a) for a in (-10.,-2.,2.,10.)]+[("stuck",0.),("noisy_stuck",.1),("quantized_stuck",.1)]
TEMPLATES = [(k,a) for k in ("bias","drift") for a in (-50.,-10.,-2.,2.,10.,50.)]+[("gain",a) for a in (-.25,-.1,.1,.25)]+[("stuck",0.)]


def materialize(raw, case, ranges):
    x=np.asarray(raw,dtype=float).copy()
    if x.shape != (300,len(ranges)): raise ValueError("Expected 300 rows in declared channel order")
    family=case.get("corruption",case["family"])
    j=case.get("target_index");a=float(case.get("amplitude",0.))
    if family in ("unchanged","native_process","native_attack"): return x
    if family == "common_mode":
        for target in case["corrupted_indices"]:
            x=materialize(x,dict(family="drift",target_index=target,amplitude=a),ranges)
        return x
    if j is None or not 0<=j<x.shape[1]: raise ValueError("Invalid corruption target")
    if family == "drift": x[:,j]+=np.clip((np.arange(300)-60)/179.,0.,1.)*a/100.*ranges[j]
    elif family == "bias": x[60:,j]+=a/100.*ranges[j]
    elif family == "gain": x[60:,j]*=1.+a
    elif family == "stuck": x[60:,j]=x[59,j]
    elif family in ("noisy_stuck","quantized_stuck"):
        rng=np.random.default_rng(int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:16],16))
        x[60:,j]=x[59,j]+rng.normal(0.,a/100.*ranges[j],240)
        if family=="quantized_stuck":
            step=.2/100.*ranges[j]
            if step<=0: raise ValueError("Quantization requires positive training range")
            x[60:,j]=np.rint(x[60:,j]/step)*step
    else: raise ValueError("Unknown corruption: "+family)
    return x


def candidates(manifest, split, *, stride=300):
    result=[]
    for source in manifest["recordings"]:
        if not source.get("normal",False): continue
        for seg in source["segments"]:
            if seg["split"]!=split: continue
            for start in range(int(seg["start"]),int(seg["end"])-299,stride):
                result.append(dict(recording_id=source["id"],start=start,end=start+300,
                                   split=split,source_sha256=source.get("sha256")))
    return sorted(result,key=lambda r:(r["recording_id"],r["start"]))


def spaced(rows,n):
    if len(rows)<n: raise ValueError(f"Insufficient source periods: {len(rows)} available, {n} required")
    return [rows[i] for i in np.linspace(0,len(rows)-1,n,dtype=int)]


def _period(unit,row,role):
    out=dict(row,dataset=unit,role=role)
    out["period_id"]=digest(out)[:24]
    return out


def source_periods(manifest,cfg):
    unit=manifest["dataset"];roles=cfg["roles"]
    # History and dictionary use the declared training role. Their exposure is explicit.
    train_pool=candidates(manifest,"train")
    train=spaced(train_pool,roles["dictionary"])
    extra_normal=spaced([r for r in train_pool if r not in train],roles.get("selection_normal_fit",0)) if roles.get("selection_normal_fit",0) else []
    val=spaced(candidates(manifest,"validation"),roles["decision_fit"]+roles["risk_calibration"])
    # Split native FourTank development by complete batch when possible.
    if unit=="four_tank":
        files=sorted({r["recording_id"] for r in candidates(manifest,"validation")})
        if len(files)<2: raise ValueError("FourTank needs distinct validation batches for fit/risk roles")
        pivot=max(1,len(files)//2)
        fit=spaced([r for r in candidates(manifest,"validation") if r["recording_id"] in files[:pivot]],roles["decision_fit"])
        risk=spaced([r for r in candidates(manifest,"validation") if r["recording_id"] in files[pivot:]],roles["risk_calibration"])
    else: fit,risk=val[:roles["decision_fit"]],val[roles["decision_fit"]:]
    # Calibration periods may overlap within the same role, as in the paper.
    cal=spaced(candidates(manifest,"calibration",stride=10),cfg["calibration_counts"].get(unit,roles["normal_calibration"]))
    assessment=spaced(candidates(manifest,"test_normal"),cfg["assessment_counts"].get(unit,roles["assessment"]))
    result=[]
    for role,rows in (("dictionary",train),("selection_normal_fit",extra_normal),("decision_fit",fit),("risk_calibration",risk),
                      ("normal_calibration",cal),("assessment",assessment)):
        result.extend(_period(unit,r,role) for r in rows)
    # Native recordings are never training input. Explicit roles supplied by the
    # adapter are retained. Otherwise whole recordings are time-ordered by role.
    native=[r for r in manifest["recordings"] if not r.get("normal",False)]
    aliases={}
    for source in sorted(native,key=lambda r:r["id"]):
        key=source.get("sha256",source["id"])
        if key in aliases: continue
        aliases[key]=source["id"]
        segments=[s for s in source.get("segments",[]) if s["end"]-s["start"]>=300]
        for index,seg in enumerate(segments):
            role=seg.get("native_role",seg.get("role",source.get("native_role",source.get("role"))))
            if role is None:
                role=("decision_fit","risk_calibration","assessment")[len(aliases)%3]
            if role not in ("decision_fit","risk_calibration","assessment"): role="assessment"
            start=min(int(seg["start"]+.70*(seg["end"]-seg["start"]))//10*10,int(seg["end"])-300)
            row=dict(recording_id=source["id"],start=start,end=start+300,split="native",
                native_category=source.get("native_category","locally_unverified"),
                qualification=seg.get("qualification",source.get("response_qualification",source.get("qualification","locally_unverified"))),
                response_supported=bool(seg.get("response_supported",source.get("response_supported",False))),
                clean_counterfactual=False,native_segment=index)
            result.append(_period(unit,row,role))
    return result


def case_for(period,family="unchanged",target=None,amplitude=0.,**extra):
    value=dict(period_id=period["period_id"],dataset=period["dataset"],role=period["role"],
               panel="primary",family=family,corruption=family,target_index=target,
               amplitude=amplitude,context_status="available")
    value.update(extra)
    value["case_id"]=digest(value)[:32]
    return value


def study_cases(periods,features,cfg):
    cases=[];demo=cfg["preset"]=="demo";d=len(features)
    bases={}
    for period in periods:
        role=period["role"]
        if period["split"]=="native":
            c=case_for(period,"native_attack" if period["dataset"]=="swat" else "native_process")
            c.update(panel="native",native_category=period["native_category"],
             response_supported=period["response_supported"],clean_counterfactual=False)
            cases.append(c);continue
        base=case_for(period);cases.append(base);bases[period["period_id"]]=base["case_id"]
        if role in ("normal_calibration","selection_normal_fit"): continue
        specs=TEMPLATES if role=="dictionary" else (FIT if role=="decision_fit" else RISK if role=="risk_calibration" else ASSESS)
        if demo and role in ("decision_fit","risk_calibration"):
            specs=[("drift",-10.),("drift",10.),("stuck",0.)]
        for j in range(d):
            for family,amplitude in specs:
                cases.append(case_for(period,family,j,amplitude,base_id=base["case_id"]))
        # Assessment controls apply on the four configurations with documented
        # primary/witness channels. All other configurations retain primary cases.
        unit=period["dataset"]
        witness={"hai_ds1":"mixer0.sensor_continuous_pressure_tank_B201.p",
          "hai_ds4":"bottling0.tank_B401.level","four_tank":"FIC23002_PV_Out ValueY","swat":"FIT401"}
        if demo: witness[unit]=features[-1]
        peers=[p for p in periods if p["role"]=="assessment" and p["split"]!="native"]
        if role=="assessment" and unit in witness and period in peers[:3]:
            w=features.index(witness[unit])
            declared={"hai_ds1":["mixer0.tank_B203.level","mixer0.sensor_continuous_pressure_tank_B203.p"],
              "hai_ds4":["bottling0.tank_B402.level","bottling0.sensor_continuous_pressure_tank_B402.p"],
              "four_tank":["LIC23002_PV_Out ValueY","LevelMainTank ValueY"],
              "swat":["LIT401","FIT401"]}
            pair=[features.index(f) for f in declared[unit]] if unit in declared else [0,1]
            for a in (-10.,10.):
                cases.append(case_for(period,"witness_corruption",w,a,panel="witness",corruption="drift"))
                cases.append(case_for(period,"common_mode",None,a,panel="common_mode",corrupted_indices=pair))
            for state in ("missing","misaligned","untrusted"):
                cases.append(case_for(period,"context_control",0,10.,panel="context_violation",corruption="drift",context_status=state))
    return cases


def temporal_periods(manifest,primary,read_frame,cfg):
    from .context import context_features
    unit=manifest["dataset"]
    if unit not in ("swat","demo") or not cfg.get("temporal"):
        return [],dict(status="not_applicable")
    eligible=[]
    for row in candidates(manifest,"test_normal"):
        if any(p["recording_id"]==row["recording_id"] and
               row["start"]<p["end"]+60 and row["end"]>p["start"]-60 for p in primary): continue
        frame=read_frame(row)
        cls=context_features(frame,manifest.get("context_unit",unit))["history_class"]
        if cls not in ("steady","transition_settling"): continue
        eligible.append(dict(row,history_class=cls))
    target=4 if cfg["preset"]=="demo" else 48 if len(eligible)>=48 else 24 if len(eligible)>=24 else 0
    if not target: return [],dict(status="skipped",reason="fewer_than_24_eligible_blocks",eligible=len(eligible))
    # Greedy deterministic time spacing, then stratified time-spaced selection.
    separated=[]
    for row in eligible:
        if separated and row["recording_id"]==separated[-1]["recording_id"] and row["start"]<separated[-1]["end"]+60: continue
        separated.append(row)
    if len(separated)<target:
        target=4 if cfg["preset"]=="demo" and len(separated)>=4 else 24 if len(separated)>=24 else 0
    if not target: return [],dict(status="skipped",reason="insufficient_embargo_separated_blocks",eligible=len(separated))
    chosen=[]
    for cls in ("steady","transition_settling"):
        pool=[r for r in separated if r["history_class"]==cls]
        n=min(len(pool),target//2)
        if n: chosen.extend(spaced(pool,n))
    remaining=[r for r in separated if r not in chosen]
    if len(chosen)<target: chosen.extend(spaced(remaining,target-len(chosen)))
    chosen.sort(key=lambda r:(r["recording_id"],r["start"]))
    faults=[]
    for cls in ("steady","transition_settling"):
        pool=[r for r in chosen if r["history_class"]==cls]
        n=min(len(pool),1 if cfg["preset"]=="demo" else 2)
        if n: faults.extend(spaced(pool,n))
    required=2 if cfg["preset"]=="demo" else 4
    remaining=[r for r in chosen if r not in faults]
    if len(faults)<required: faults.extend(spaced(remaining,required-len(faults)))
    periods=[]
    for row in chosen:
        period=_period(unit,row,"assessment");period["temporal_fault_block"]=row in faults
        periods.append(period)
    return periods,dict(status="frozen",source_exposure="within previously exposed normal recording; fresh-model test partition",eligible=len(separated),selected=target,
            strata=dict(Counter(r["history_class"] for r in chosen)))


def temporal_cases(periods,d):
    result=[]
    for p in periods:
        result.append(case_for(p,panel="temporal"))
        if p["temporal_fault_block"]:
            for j in range(d):
                for family,a in TEMPORAL: result.append(case_for(p,family,j,a,panel="temporal"))
    return result
