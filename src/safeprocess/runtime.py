"""Atomic identities, stage receipts and sealed-prediction integrity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np

SCHEMA = "safeprocess-run-v1"

def safe_json(value):
    if isinstance(value, dict): return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray): return safe_json(value.tolist())
    if isinstance(value, np.generic): return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value

def digest(value):
    return hashlib.sha256(json.dumps(safe_json(value),sort_keys=True,
          separators=(",",":"),allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(safe_json(value),indent=2,sort_keys=True,
                             allow_nan=False)+"\n")
    tmp.replace(path)

def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w") as f:
        for row in rows: f.write(json.dumps(safe_json(row),sort_keys=True,
                                   allow_nan=False)+"\n")
    tmp.replace(path)

def utc(): return datetime.now(timezone.utc).isoformat()

def implementation_identity():
    package=Path(__file__).resolve().parent
    return digest({str(p.relative_to(package)):file_hash(p) for p in sorted(package.rglob("*"))
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".pyo")})


def open_run(work_dir, config=None):
    root=Path(work_dir).resolve()
    # Never write fresh outputs inside the code/results distribution.
    repository=Path(__file__).resolve().parents[2]
    if ((repository/"pyproject.toml").is_file() and
        (repository/"src/safeprocess").is_dir() and root.is_relative_to(repository)):
        raise ValueError("Fresh work-dir must be outside the code/results repository")
    root.mkdir(parents=True,exist_ok=True)
    path=root/"run.json"
    if path.exists():
        value=read_json(path)
        if config is not None and digest(config)!=value["config_sha256"]:
            raise ValueError("Resume configuration differs from frozen run bindings")
        if value.get("implementation_sha256") != implementation_identity():
            raise ValueError("Resume implementation differs from frozen run bindings")
    else:
        if config is None: raise ValueError("Run is absent. Start with describe/run/demo")
        value=dict(schema=SCHEMA,run_id="fresh-"+uuid.uuid4().hex,
          created_utc=utc(),config=config,config_sha256=digest(config),stages={},
          implementation_sha256=implementation_identity(),
          scientific_scope="fresh training; no exact archived-count reproduction")
        write_json(path,value)
    return root,value

def receipt(root, stage, files, **details):
    value=read_json(root/"run.json")
    hashes={str(Path(f).relative_to(root)):file_hash(f) for f in files}
    entry=dict(completed_utc=utc(),files=hashes,**safe_json(details))
    old=value["stages"].get(stage)
    if old and old["files"]!=hashes:
        raise ValueError("Completed stage changed: "+stage)
    value["stages"][stage]=old or entry
    write_json(root/"run.json",value)
    return entry

DEPENDENCIES={"prepare":["describe"],"train":["prepare"],"calibrate":["train"],
 "predict":["calibrate"],"evaluate":["predict"],"ablate-history":["evaluate"],
 "report":["evaluate"],"verify":["report"]}

def require(root,stage):
    for predecessor in DEPENDENCIES.get(stage,[]): require(root,predecessor)
    value=read_json(root/"run.json")
    if stage not in value["stages"]: raise ValueError("Required stage is incomplete: "+stage)
    for rel,h in value["stages"][stage]["files"].items():
        if file_hash(root/rel)!=h: raise ValueError("Bound file changed: "+rel)
    return value["stages"][stage]
