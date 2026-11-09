# Computational source map

The standalone package uses ordinary Python imports and explicit dependencies. It requires no thesis checkout. These paths identify extraction provenance and are not runtime acquisition paths.

| Standalone implementation | Retained source / semantics |
| --- | --- |
| `models.py` | Projected LSTM AE from `anomaly_detection/models/tto.py`, source commit `09a40f56ccdbbfde079abf59327b508bf7e9a99c` |
| `training.py` | Training-only scaler and training/resume conventions from `anomaly_detection/training.py`, with one explicitly versioned fresh recipe |
| `numerics.py` | Prospective masked input optimization from `anomaly_detection/imputation.py`; evidence geometry from `attribution_probe.py` and `prominence_panel.py` |
| `data.py`, `datasets/`, `ontology.py` | Minimal local adapters, authored channel/topology metadata and CPSWatch 1.0.1 RDF. No external reasoner or ontology service |
| `kernels/relative.py` | Original calibrated event score, twelve candidate features, confidence-gap risk rules from `final_decision.py` |
| `kernels/ranker.py` | Portable linear/tree scorer convention from `final_calibrator.py` |
| `kernels/history.py` | Shared-history profile and admission from `history_core.py` |
| `context.py` | Component-history ridge centre/log-scale calibration from `context_refinement.py` |
| `kernels/selection.py` | Pooled absolute scorer, weighting and exactly-one-valid selection from `selection_refinement.py` |
| `kernels/prominence.py`, `kernels/dominance.py` | Residual/Jacobian process graph evidence and dominance from `prominence_panel.py` / `residual_drift.py` |
| `signatures.py` | Signed temporal dictionary and envelope functions from `signature_panel.py` |
| `scenarios.py` | Versioned portable generators and source-role planner using the paper's fault matrix and controls |
| `calibration.py`, `policies.py` | Standalone fitting and observable-only seven-policy orchestration, with explicit insufficient-calibration receipts |
| `runtime.py`, `workflow.py`, `evaluation.py`, `reporting.py`, `cli.py` | New local stage runner and reporting tools for fresh identities, bound resume and prediction/evaluator separation |

The seven methods are Original calibrated, residual dominance, envelope screening, Reference, Context, Selection and Combined. The thesis supervisor has a different comparison and is not imported or exposed here.

Direct implementation checks compare TTO with the retained source and policy/context results with 31 closed numerical fixtures spanning all twelve configurations. These checks establish extraction agreement at the tested interfaces. The bounded synthetic run establishes executable stage integration. Neither replaces a full fresh-data campaign or claims fresh-score equality with the reported results.

The paper, exact method specification and fixed evidence retain their own citations and provenance. The source schema's CC BY 4.0 notice and the official IFAC formatting notices remain applicable alongside the repository's Apache 2.0 license.

## Reconstruction references

The fixed-network sensor-reconstruction principle is described in Ludwig, Ehrhardt and Niggemann, [*Creating Virtual Sensors Using Neural Networks*](https://doi.org/10.1109/ETFA65518.2025.11205672), IEEE ETFA 2025. The TTO formulation and missing-to-observed coupling analysis are described in Ludwig, Windmann, Heesch and Niggemann, [*From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels*](https://arxiv.org/abs/2611.06612), arXiv preprint, 2026 (v1 submitted 4 November).

These references support the reconstruction principle and its formulation. The exact SafeProcess returned-estimate, stopping, normalization, scoring and routing settings remain separately specified. Historical source snapshots retain their dated provenance records.
