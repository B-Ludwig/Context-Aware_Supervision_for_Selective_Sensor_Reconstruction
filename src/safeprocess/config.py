"""Versioned fresh-run protocol; independent of the supplied results artifact."""
from copy import deepcopy
import json
from pathlib import Path

DATASETS = [f"hai_ds{i}" for i in range(1, 11)] + ["four_tank", "swat"]
SEEDS = [23, 42, 101, 2027, 31415]
TRAINING = dict(input_proj_dim=64, hidden_size=128, latent_fraction=.5,
 latent_cap=16, n_layers=1, post_mlp_ratio=4, dropout=.1, post_mlp_dropout=0.,
 batch_size=512, max_epochs=3000, lr=.001, patience=100, min_delta=.001,
 grad_clip=1., window_length=60, stride=10, weight_decay=0.)
TTO = dict(max_cycles=5000, init_window=100, patience=100,
 max_consecutive_cycles=200, learning_rate=.01, smooth_lambda=.001,
 max_seconds=60.)
ASSESSMENT_COUNTS = dict(zip(DATASETS, [10,10,10,10,11,12,11,11,12,11,12,12]))
CALIBRATION_COUNTS = dict(zip(DATASETS, [7,6,6,7,7,8,7,7,8,7,12,24]))
METHODS = ["original_calibrated", "residual_dominance", "envelope_screened",
           "reference", "context", "selection", "combined"]

def resolve_config(name="paper", datasets=None, seeds=None):
    if name in ("paper", "demo"):
        cfg = dict(schema_version=1, protocol="safeprocess-fresh-v1", preset=name,
          datasets=DATASETS.copy(), seeds=SEEDS.copy(), training=deepcopy(TRAINING),
          tto=deepcopy(TTO), temporal=True, history_ablation=True,
          roles=dict(dictionary=8, selection_normal_fit=8, normal_calibration=24, decision_fit=4,
                     risk_calibration=3, assessment=12),
          assessment_counts=ASSESSMENT_COUNTS, calibration_counts=CALIBRATION_COUNTS,
          temporal_seeds=[23,42])
        if name == "demo":
            cfg.update(datasets=["demo"], seeds=[23], roles=dict(dictionary=4,
             normal_calibration=3, selection_normal_fit=2, decision_fit=1, risk_calibration=1, assessment=1),
             assessment_counts={"demo":1}, calibration_counts={"demo":3},
             temporal_seeds=[23])
            cfg["training"].update(input_proj_dim=8, hidden_size=8, latent_cap=2,
              batch_size=128, max_epochs=5, patience=5)
            cfg["tto"].update(max_cycles=500, init_window=5, patience=10,
              max_consecutive_cycles=20, max_seconds=10.)
    else:
        cfg=json.loads(Path(name).read_text())
        if cfg.get("schema_version") != 1:
            raise ValueError("Unsupported configuration schema")
    if datasets:
        unknown=set(datasets)-set(cfg["datasets"])
        if unknown: raise ValueError(f"Datasets absent from configuration: {unknown}")
        cfg["datasets"]=list(datasets)
    if seeds:
        if any(int(s) not in SEEDS for s in seeds):
            raise ValueError(f"Supported seeds: {SEEDS}")
        cfg["seeds"]=[int(s) for s in seeds]
    # Seed 23 alone fits the decision model. Transfer it unchanged to other seeds.
    cfg["development_seed"]=23
    return cfg
