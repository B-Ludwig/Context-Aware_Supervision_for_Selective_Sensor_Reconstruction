# Fresh training and study workflow

The installable implementation runs independently of the thesis checkout and the supplied `results/`. The package ships the ontology schema, authored mappings and a runtime synthetic generator. It can download FourTank recordings from their pinned provider archive and import HAI recordings from a provider archive or supplied download URL. SWaT recordings require user access. No ontology server is contacted.

## Install and run

Use Python 3.10 or newer. The recorded implementation checks use Python 3.10 and the versions in [fresh-tested.txt](../requirements/fresh-tested.txt). A CUDA-compatible PyTorch installation is needed only when explicitly selecting CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
safeprocess demo --work-dir /tmp/safeprocess-demo

safeprocess describe --config paper --data-root /path/to/recordings \
  --work-dir /tmp/safeprocess-preflight --preflight
safeprocess run --config paper --data-root /path/to/recordings \
  --work-dir /path/to/runs/fresh-paper --device cuda
```

Choose a new work directory outside the checkout and the supplied results. CPU is the default and remains supported. The full paper matrix is computationally substantial. The demo changes model size, training duration, TTO limits and cohort size to exercise the implementation quickly. Its scores are not scientific evidence for the paper.

For an incremental dataset/checkpoint run:

```bash
safeprocess run --config paper --datasets swat --seeds 23 42 \
  --data-root /path/to/recordings --work-dir /path/to/runs/fresh-swat --device cuda
```

Seed 23 fits the decision scorer and thresholds even when only another assessment seed is selected. Those rules transfer unchanged to the other checkpoints. Normal residual profiles bind to each checkpoint's numerical evidence. The pooled Selection scorer is fitted across the configurations included in this fresh run. A subset run therefore has a distinct selector identity from a full twelve-configuration run.

## Ordered stages

Every stage is an executable command. Following `describe`, the work directory's resolved configuration is reused when `--config` is omitted. `fetch-data` only needs `--data-root`, without a run work directory. See the [acquisition guide](DATA_ACQUISITION.md) for provider links, HAI archive import and SWaT access.

| Command | Reads and writes |
| --- | --- |
| `fetch-data` | Downloads the pinned public FourTank archive, imports/downloads a supplied HAI archive, or reports the recordings that require user access. It performs no training or inference |
| `describe` | Acquires missing inputs for a new run, converts recordings, preserves source identities and partitions, writes ordered sensor/context/owner mappings, offline validated RDF and manifest |
| `prepare` | Freezes deterministic source periods, development roles, corruptions and assessment inventory. Writes corrupted observations, context-only files and separately stored evaluator references |
| `train` | Fits train-only min–max scaling and fresh AEs. Exports best checkpoint, scaler, normal-calibration native threshold and resume state |
| `calibrate` | Computes normal/development residuals, absolute input Jacobians and masked candidate reconstructions. Fits dictionary, normalizers, guards, relative scorer and pooled absolute scorer. Freezes every profile before prediction |
| `predict` | Receives only corrupted sensor observations, declared operating context and frozen profiles. Shares one numerical payload across seven policies. Retains successful, invalid, abstained, timed-out and failed terminal outcomes, then closes predictions |
| `evaluate` | Verifies closure, joins evaluator-only clean references and injected targets, writes coverage, harm, signed error and checkpoint/configuration/family/source-period summaries |
| `ablate-history` | Uses numerical caches on CPU. Fits four separately identified auxiliary normalizers and reports risk-refitted and fixed-full-Context-gap comparisons for each selected checkpoint. Missing caches produce an unavailable receipt, never inference |
| `report` | Writes a coverage–harm figure, report and deterministic illustrative trace catalog, including retained context, centre and scale trajectories |
| `verify` | Checks bound predecessor files, seven-policy terminal coverage, one-to-one evaluator joins, missing native counterfactuals and prediction-closure chronology |
| `run` | Executes these stages in order |
| `demo` | Executes the same stages on small runtime-generated synthetic inputs and a bounded CPU training recipe |

Example stage execution:

```bash
safeprocess describe --config paper --datasets hai_ds1 --seeds 23 \
  --data-root /path/to/recordings --work-dir /path/to/runs/fresh-hai
safeprocess prepare --work-dir /path/to/runs/fresh-hai
safeprocess train --work-dir /path/to/runs/fresh-hai --device cpu
safeprocess calibrate --work-dir /path/to/runs/fresh-hai --device cpu
safeprocess predict --work-dir /path/to/runs/fresh-hai --device cpu
safeprocess evaluate --work-dir /path/to/runs/fresh-hai
safeprocess ablate-history --work-dir /path/to/runs/fresh-hai
safeprocess report --work-dir /path/to/runs/fresh-hai
safeprocess verify --work-dir /path/to/runs/fresh-hai
```

## Versioned fresh-training recipe

`safeprocess-fresh-projlstm-v1` uses projection 64, one encoder and decoder LSTM layer of hidden size 128, and a residual MLP of width 512. The timewise bottleneck is `max(1, min(16, floor(0.5 * channels)))`. Recurrent dropout is inactive for one layer. Post-MLP dropout is zero.

Adam uses learning rate 0.001, batch size 512, gradient norm clipping 1 and zero weight decay. Training stops after at most 3,000 epochs or 100 epochs without relative validation improvement of 0.001. Independently, every lowest validation loss updates the best checkpoint. There is no masked/noisy training or mixed precision. All model inputs are length-60 windows with stride 10. The scaler uses each training row once, never validation/calibration/test values. Seeds are 23, 42, 101, 2027 and 31415.

This is a uniform fresh-training configuration. Its weights do not claim to reproduce archived checkpoints or their exact reported counts. The [configuration exports](../configs/) and each run's `run.json` expose the resolved settings. Custom experiments may use a copy of that JSON, with a new work directory and identity. They do not alter supplied evidence.

TTO starts masked values at the training mean. Adam optimizes the normalized observed-input error plus smoothness, with input-gradient L2 normalization and frozen model weights. The production settings are 5,000 maximum cycles, learning rate 0.01, smoothness 0.001, bootstrap 100 cycles, patience 100, plateau limit 200 and 60 seconds per candidate batch. The relative improvement limit is 0.01 times the mean bootstrap loss. The returned estimate is the AE output at the final optimized masked inputs. It preserves observed cells. Budget exhaustion alone is not convergence. Invalid candidates remain invalid. A nonfinite candidate receives a versioned observable placeholder solely to keep other policies computable. Its failure and invalid status are retained. The absolute scorer keeps the defined cross-candidate score gap while excluding that invalid candidate from qualification. A baseline proposal selecting an invalid estimate has missing reconstruction quality, and its injected-target error remains the observed error. No placeholder is scored as a reconstruction.

## Source roles and scenarios

Selection uses time-spaced source coordinates without consulting model outcomes. Dictionary and normal-reference evidence share eight explicitly training-exposed periods. Normal-only Selection fitting uses eight additional nonoverlapping training periods. Validation supplies four decision-fit and three risk-calibration periods. Normal calibration and normal assessment use their designated partitions and versioned per-configuration counts. Calibration periods may overlap within their own role. No period crosses a recording, batch or split boundary. The role map is stored before inference.

The dictionary uses all 17 fixed template settings per channel and base period: signed bias/drift at 2%, 10% and 50%, four gains and exact stuck. Decision fitting and risk calibration use their distinct signed-drift grids and exact-stuck cases. Only declared development roles supply scorer labels. A positive label requires a valid, correct drift/exact-stuck target with lower clean-reference MAE. Other candidates and controls are negative. An unfittable or infeasible learner produces an explicit ineligible receipt and abstention, never a fabricated model.

The primary assessment applies twelve signed drifts and exact/noisy/quantized stuck per channel. There are paired unchanged controls. The declared witness and common-mode panels use the four documented configurations, with signed 10% drift. Missing, misaligned and untrusted-context controls test the context contract. Native source categories retain provider provenance and explicit qualification, including unknown local response. See [DATA_ACQUISITION.md](DATA_ACQUISITION.md). No residual-derived response qualification is created.

The fresh SWaT temporal panel enumerates 300-row normal-test blocks, excludes the fresh primary footprints plus a 60-row embargo, separates blocks by at least 60 rows and classifies histories causally. It selects 48 blocks, aiming for 24 per stratum, falling back to 24 blocks or a skip receipt. Four fault blocks each contribute eleven channels times four signed drifts (2%, 10%) and three stuck variants. With 48 unchanged blocks this is 356 observations per checkpoint, using selected seeds 23 and 42. A stratum shortfall is filled from the other stratum. This fresh panel does not reproduce the archived temporal cohort or assert that its source intervals escaped every historical campaign. Its same-recording exposure is explicit.

## Prediction boundary, error accounting and resume

`plans/observable.jsonl` contains no injected targets, families, clean values or quality labels. Its context files contain only declared state/actuator columns. The policy API accepts numerical residual/candidate evidence, these context columns, fixed profiles and ontology bindings. Clean sensor references live under `evaluator/periods/`. Assessment labels are joined only after `predictions/closed.json` has bound every terminal record and payload. Development labels are confined to calibration and explanatory auxiliary fitting.

A selected-channel signed benefit is observed MAE minus reconstructed MAE, divided by that channel's training range. Abstention contributes zero when a quality reference exists. A wrong-target proposal retains damage on the selected channel, and the injected target's remaining error is reported separately. Common-mode selected-channel improvement does not imply complete recovery. Native counterfactual quality remains missing. Terminal failures remain in case denominators. Unchanged controls have their own panel, separate from primary fault coverage.

The primary summaries pool the configured case matrix. Separate configuration, checkpoint, fault-family and source-period tables expose denominators and dependence. These synthetic-case frequencies are not deployment prevalence. The auxiliary ladder is explanatory: its normalization can change event admission even with a fixed confidence gap.

Resume uses the same command and work directory. Configuration and implementation hashes must match. All completed predecessor bindings are verified. Training restores its optimizer, patience, random generator and best checkpoint. Cached numerical inference and terminal assessment failures are immutable. An unfinished observation may resume, but a terminal failure is never retried and cases are never replaced after viewing outcomes. Changing source data, code or configuration requires a new run.

## Outputs and reproduction scope

Each fresh work directory contains resolved configuration and run identity, converted sources, ontology descriptions, source-role plans, model/scaler identities, fixed profiles, numerical caches, closed predictions, evaluation, history ladder, figures and integrity receipts. It is self-contained for resuming and CPU reporting after preparation. Do not add it to Git.

The repository's supplied historical results use the existing commands in the root README and [RESULTS.md](RESULTS.md). Those commands still reproduce the paper's closed tables and figures without fresh model execution. Fresh runs write outside that distribution. Public Pipeline APIs, the diagnosis reasoner, plant actions, NoBoom, S3/DVC services, model search and thesis orchestration are excluded from the package.
