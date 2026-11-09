"""CPU reports from closed predictions. No training or policy selection."""
import pandas as pd
from .runtime import read_json,read_rows,write_json


def make_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","pdf.fonttype":42})
    out=root/"report";out.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(root/"evaluation"/"summary.csv")
    primary=table[table.panel=="primary"]
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    labels=primary.method.str.replace("_"," ")
    axes[0].barh(labels,primary.beneficial_coverage)
    axes[0].set_xlabel("Correct-beneficial coverage")
    axes[1].barh(labels,primary.harm_per_quality_proposal.fillna(0),color="#b34b40")
    axes[1].set_xlabel("Strict harm / quality-evaluable proposals")
    fig.tight_layout();figure=out/"coverage_harm.pdf";fig.savefig(figure);plt.close(fig)
    join=read_json(root/"evaluation"/"join.json")
    text=["# Fresh SafeProcess run", "", f"Run: `{read_json(root/'run.json')['run_id']}`", "",
      "These results come from the versioned fresh-training recipe. They do not replace the manuscript's archived checkpoints or reported results.","",
      "The figure pairs useful coverage with strict proposal harm. Native clean counterfactuals remain missing. Failures stay in observation denominators.","",
      "Pooled summaries weight the configured scenario matrix. Configuration, checkpoint, fault-family and source-period tables expose dependence and concentration of effects.","",
      f"Closed observations: {join['observations']}. Seven-policy outcome rows: {join['rows']}.","",
      "See `../evaluation/summary.csv`, `by_configuration.csv`, `by_checkpoint.csv`, `by_fault_family.csv` and `by_source_period.csv` for exact denominators.","",
      "The temporal panel uses separate blocks within the same normal recording. Source exposure is recorded in `../plans/temporal.json`.","",
      "The history ladder is explanatory and uses development roles for auxiliary fitting. Both its threshold-refitted and fixed-full-Context-gap comparisons can change event admission. Neither isolates a causal history-feature effect.","",
      "No deployment prevalence, significance test or assessment-selected replacement operating point is inferred.",""]
    report=out/"REPORT.md";report.write_text("\n".join(text))
    return [figure,report]+make_traces(root,out)


def make_traces(root,out):
    """Deterministic illustrative selections; unavailable classes stay explicit."""
    import numpy as np
    import matplotlib.pyplot as plt
    from .context import context_features,conditional_normalizer
    rows=[r for r in read_rows(root/"evaluation"/"outcomes.jsonl")
          if r["method"]=="context" and r["status"]=="completed"]
    chosen={}
    positive=[r for r in rows if r["correct_beneficial"]]
    adverse=[r for r in rows if r["strict_harm"]]
    veto=[r for r in rows if r["preview"]]
    native=[r for r in rows if r["panel"]=="native"]
    if positive: chosen["useful"]=min(positive,key=lambda r:(-r["signed_benefit"],r["case_id"],r["seed"]))
    if adverse: chosen["adverse"]=min(adverse,key=lambda r:(r["signed_benefit"],r["case_id"],r["seed"]))
    if veto or native: chosen["veto_or_native"]=min(veto or native,key=lambda r:(r["case_id"],r["seed"]))
    obs={r["case_id"]:r for r in read_rows(root/"plans"/"observable.jsonl")}
    catalog=dict(illustrative=True,rule="Context only. Useful: greatest positive correct-target signed benefit. Adverse: most negative selected-channel benefit. Veto: preview case. If no preview exists use native case. Ties use lexical case ID then seed. Missing classes are unavailable.",cases={})
    files=[]
    for label in ("useful","adverse","veto_or_native"):
        if label not in chosen:
            catalog["cases"][label]=dict(status="unavailable",reason="no_matching_closed_outcome");continue
        row=chosen[label];unit=row["dataset"];seed=row["seed"];case=row["case_id"]
        terminal=read_json(root/"predictions"/unit/f"seed-{seed}"/(case+".json"))
        pred=terminal["predictions"]["context"]
        with np.load(root/terminal["payload"],allow_pickle=False) as z: payload={k:z[k] for k in z.files}
        with np.load(root/"evaluator"/"periods"/unit/(row["period_id"]+".npz"),allow_pickle=False) as z: clean=z["clean"]
        profile=read_json(root/"profiles"/unit/f"seed-{seed}.json")
        manifest=read_json(root/"datasets"/unit/"manifest.json")
        frame=pd.read_parquet(root/obs[case]["context_path"]).loc[:,manifest["context_columns"]]
        context=context_features(frame,manifest["context_unit"],obs[case]["context_status"])
        center,scale,flags=conditional_normalizer(context,profile["context"])
        j=pred.get("target")
        if j is None: j=pred.get("preview")
        if j is None: j=pred.get("ranked_target")
        if j is None: j=0
        j=int(j)
        fig,axes=plt.subplots(3,1,figsize=(7,5.3),sharex=False)
        axes[0].plot(payload["raw"][:,j],label="observed",linewidth=1)
        if row["family"] not in ("native_process","native_attack"):
            axes[0].plot(clean[:,j],label="clean reference",linewidth=1)
        if payload["valid"][j]:
            axes[0].plot(np.arange(240,300),payload["corrected"][j,:,j],label="candidate",linewidth=1)
        axes[0].set_ylabel(manifest["features"][j]);axes[0].legend(fontsize=8)
        ends=np.array([259,269,279,289,299])
        axes[1].plot(ends,center[:,j],marker="o");axes[1].set_ylabel("residual centre")
        axes[2].plot(ends,scale[:,j],marker="o");axes[2].set_ylabel("residual scale");axes[2].set_xlabel("source-relative row")
        fig.tight_layout();path=out/("trace_"+label+".pdf");fig.savefig(path);plt.close(fig);files.append(path)
        data=out/("trace_"+label+".json")
        write_json(data,dict(case_id=case,dataset=unit,seed=seed,channel=j,context=context,
                            center=center,scale=scale,fallback_flags=flags,prediction=pred,outcome=row))
        files.append(data);catalog["cases"][label]=dict(status="available",case_id=case,seed=seed,figure=path.name,numerical=data.name)
    path=out/"trace_selection.json";write_json(path,catalog);files.append(path)
    return files
