"""Public command line; ordinary imports and local user-supplied recordings."""
import argparse
import json
import os
from pathlib import Path
from .config import resolve_config
from .runtime import open_run,read_json

STAGES=("describe","prepare","train","calibrate","predict","evaluate","ablate-history","report","verify")

def main(argv=None):
    parser=argparse.ArgumentParser(prog="safeprocess",description="Standalone fresh-training and closed-evidence study workflow")
    sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch-data",help="Download public FourTank data and import a HAI provider archive")
    fetch.add_argument("--data-root",type=Path,required=True)
    fetch.add_argument("--datasets",nargs="+",default=["hai","four_tank"],
                       help="hai, individual hai_ds1..10, four_tank or swat")
    fetch.add_argument("--hai-archive",type=Path,help="Local provider ZIP containing continuous CSV recordings and sim_setup.json")
    fetch.add_argument("--hai-url",help="Direct HTTPS provider archive URL obtained through IEEE DataPort")
    for name in (*STAGES,"run","demo"):
        p=sub.add_parser(name)
        p.add_argument("--work-dir",type=Path,required=True)
        p.add_argument("--config",default=None,help="paper, demo, or a versioned JSON configuration")
        p.add_argument("--data-root",type=Path,default=None)
        p.add_argument("--datasets",nargs="+",help="Subset, e.g. hai_ds1 swat")
        p.add_argument("--seeds",nargs="+",type=int)
        p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
        p.add_argument("--threads",type=int,default=1)
        if name in ("describe","run"):
            p.add_argument("--hai-archive",type=Path,help="Import the HAI recording archive before preparation")
            p.add_argument("--hai-url",help="Direct HTTPS HAI provider archive URL")
            p.add_argument("--no-download",action="store_true",help="Use existing recordings only")
        if name=="describe": p.add_argument("--preflight",action="store_true",help="Check real acquisition schemas without conversion or inference")
    args=parser.parse_args(argv)
    if args.command=="fetch-data":
        return acquire_inputs(args.data_root,args.datasets,hai_archive=args.hai_archive,hai_url=args.hai_url)
    if args.threads<1: parser.error("--threads must be positive")
    os.environ.setdefault("OMP_NUM_THREADS",str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS",str(args.threads))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if args.device=="cpu": os.environ["CUDA_VISIBLE_DEVICES"]=""
    work=args.work_dir.resolve();existing=(work/"run.json").exists()
    if existing and args.config is None and args.datasets is None and args.seeds is None:
        cfg=read_json(work/"run.json")["config"]
    else:
        cfg=resolve_config(args.config or ("demo" if args.command=="demo" else "paper"),args.datasets,args.seeds)
    if args.command=="demo" and cfg["preset"]!="demo": parser.error("demo requires the demo configuration")
    if getattr(args,"preflight",False):
        from .data import preflight_dataset
        if args.data_root is None: parser.error("--data-root is required for preflight")
        checks=[preflight_dataset(u,args.data_root) for u in cfg["datasets"]]
        work.mkdir(parents=True,exist_ok=True)
        from .runtime import write_json
        write_json(work/"preflight.json",checks)
        print(json.dumps(checks,indent=2));return 0
    if args.command in ("run","describe") and not existing and cfg["preset"]!="demo":
        if args.data_root is None: parser.error("--data-root is required for describe/run")
        if not args.no_download:
            code=acquire_inputs(args.data_root,cfg["datasets"],hai_archive=args.hai_archive,hai_url=args.hai_url)
            if code: return code
    root,manifest=open_run(work,cfg)
    from . import workflow
    import torch
    torch.set_num_threads(args.threads)
    ordered=STAGES if args.command in ("run","demo") else (args.command,)
    for stage in ordered:
        print(f"[{manifest['run_id']}] {stage}",flush=True)
        function=getattr(workflow,stage.replace("-","_"))
        if stage=="describe":
            if not existing and args.data_root is None and cfg["preset"]!="demo": parser.error("--data-root is required for describe/run")
            function(root,cfg,args.data_root or root/"synthetic-inputs")
        elif stage in ("train","calibrate","predict"):function(root,cfg,args.device)
        else:function(root,cfg)
    print(str(root/"run.json"));return 0


def acquire_inputs(data_root,datasets,*,hai_archive=None,hai_url=None):
    """Acquire source files before freezing any fresh run or loading a model."""
    from .acquisition import acquire_datasets
    from .runtime import write_json
    try:
        result=acquire_datasets(data_root,datasets,hai_archive=hai_archive,hai_url=hai_url)
    except (OSError,ValueError) as exc:
        print("Dataset acquisition failed: "+str(exc))
        return 2
    path=Path(data_root).expanduser().resolve()/"acquisition.json"
    write_json(path,result)
    for name,provider in result["providers"].items():
        print(name+": "+provider["status"])
        if provider.get("message"): print(provider["message"])
    print("Acquisition receipt: "+str(path))
    if not result["complete"]:
        print("Provide the remaining recordings or a HAI provider archive, then rerun the command. No model training was started.")
        return 2
    return 0
