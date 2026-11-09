# Method specification

The section identifiers MS-01–MS-10 match the paper's method locators. This document describes the numerical rules. The standalone package implements them without a thesis checkout or ontology service.

## MS-01 — Scope and implementation

The reported primary comparison covers HAI-CPPS DS1–DS10, FourTank and the eleven-channel SWaT subset, across seeds 23, 42, 101, 2027 and 31415. There are 193,350 primary faults and 1,335 controls per policy. The temporal panel is separate, with 616 faults and 96 unchanged observations per policy.

The seven policies are Original calibrated, Residual dominance, Envelope screened, Reference, Context, Selection and Combined. Native-alarm retention and previews compose concern with Context without changing its proposals.

- `src/safeprocess/context.py`: causal context, transforms and conditional normalizers.
- `src/safeprocess/calibration.py`: reference/development fitting and threshold selection.
- `src/safeprocess/kernels/`: event, ranking, absolute-selection, dominance and graph rules.
- `src/safeprocess/numerics.py`: model evidence and TTO.
- `src/safeprocess/policies.py`: observable decision composition.
- `configs/paper.json`: the versioned fresh-training recipe and scenario configuration.

The supplied outcomes reproduce the reported counts. Fresh training produces new weights and profiles, rather than promising the same counts. Replay profiles under `results/replay/` support the supplied fixtures and matched example. They are not a complete deployment model distribution.

## MS-02 — Observations and operating context

An observation has 300 rows. Five length-60 scoring windows start at offsets 200, 210, 220, 230 and 240. Each window uses only history available by its end. Candidate reconstruction concerns the final 60 rows. Row coordinates do not imply a common physical sampling interval across datasets.

HAI uses recorded binary component-state columns containing `.state_` and ending `.active`, grouped by component prefix. Each row requires exactly one active state per component. FourTank uses the integer controller step `CuStepNo ValueY`. SWaT uses the six actuator codes `MV301,MV302,P301,P302,P401,P402`, separate from the AE's sensor inputs and attack labels. The context source is assumed trustworthy.

Each component contributes its current state, per-state window occupancy, within-window change count, age since the latest available transition and an age-censored bit. Without an observed transition, age is the prefix length. Joint features summarize system-wide change count and age, and whether any transition has occurred. Components and reference states are sorted.

The full feature order is joint change count and age, then each component's change count, age and censor bit, followed by joint steady/transition indicators and each component's current-state one-hots and occupancy fractions. Numeric transforms use normal-reference mean and population standard deviation, floored at 1. Categories and transforms are fitted within each training fold during out-of-fold fitting.

Unknown states produce support/fallback flags. Missing, nonfinite, inconsistent or explicitly untrusted context produces invalidity. Support and validity remain separate. Sensor order and ownership are declared in `src/safeprocess/resources/paper_channels.json` and the dataset descriptions.

## MS-03 — Residual calibration and events

Write signed scaled residuals as `e = observed − AE(observed)`.

**Original calibrated** pools a signed median centre and RMS about that centre, floored at 1e-6. A dominant-phase group needs two contexts and three windows. Its centre and scale mix 80% group and 20% pooled values. A phase occupies at least 48 of a window's 60 rows, otherwise the phase is transition. The event combines: third-largest window mean-channel squared residual divided by the native threshold, largest channel's third-largest standardized energy, and largest absolute cumulative signed-window mean divided by the square root of prefix length. Dictionary higher-q95 scales each component. Separate normal calibration supplies the higher-q95 of the maximum component. Phase limits require four calibration contexts, otherwise pooling applies. Exceedance is strict.

**Reference** pools a signed median centre and RMS scale. For a joint steady/transition class with `n` reference periods, its centre shrinks with weight `n/(n+8)`. Its scale is the square root of the corresponding weighted group and pooled variances. Without class support it pools. The joint class applies to the whole observation. Separate class-specific event limits require four calibration periods.

**Context** fits an intercept ridge regression with penalty 10 for each channel's expected window residual mean. A second ridge model predicts log RMS error about out-of-fold centre estimates. Source-period GroupKFold uses up to five folds, with all five windows kept together and fold-local transforms. With fewer than two reference groups, conditional fitting is unavailable and pooled normalization applies. Predicted scale is bounded to 0.25–4 times pooled scale. Unknown or invalid window context uses pooled numerical normalization.

For Reference and Context, `z=(e−centre)/scale`, `q[w,j]=mean_t(z²)` and `m[w,j]=mean_t(z)`. The event score is the maximum of the channelwise third-largest `q` across windows and `max_{j,k}(sum_{w<=k}m[w,j])²/k`. An event strictly exceeds the empirical normal-calibration higher-q95. Legitimate transitions remain in calibration.

Context low support means calibration count below 20, class calibration count below four, reference-class count below eight, reference-group count below five, or an unknown state. It does not by itself invalidate context. The main study does not demonstrate unseen-state performance.

## MS-04 — Candidate ranking and acceptance

Each channel is masked separately in the final window. The signature dictionary bins signed errors into six consecutive ten-row bins per window. Each bin is whitened before flattening in window/bin/channel order. Covariance shrinkage is `0.8*cov + 0.2*diag(diag(cov))`, with eigenvalue floor `max(max_eigenvalue*1e-6,1e-12)`. Templates comprise signed bias/drift at 2%, 10% and 50% of training range, gains −0.25, −0.10, 0.10 and 0.25, and exact stuck. Least-squares positive template amplitude is clipped to [0.1,4]. Candidate distance is the minimum template/anchor mean squared error.

For candidate `j`, let `a_j` be signature distance, `n=max(normal_distance,1e-12)`, `b_j` own final-window MSE, `c_j` mean absolute correction and `g_j` the fractional reduction in other-channel MSE. Let `r_j` be maximum repaired-channel MSE divided by reference MSE, `l_j` the repaired-envelope limit, `h_j` normal reconstruction MAE and `s_j` training standard deviation.

| Index | Serialized feature | Definition |
|---|---|---|
|0|log_normal_distance|log1p(n), constant across candidates|
|1|relative_signature_fit|clip(a_j/n,0,100)|
|2|signature_improvement|clip((n−a_j)/n,−100,1)|
|3|signature_margin|clip((min_{k≠j}a_k−a_j)/n,−100,100)|
|4|own_residual_share|b_j/max(sum b,1e−20)|
|5|cross_gain|clip(g_j,−100,1)|
|6|log_repaired_residual_ratio|log1p(r_j/max(l_j,1e−12))|
|7|log_change_to_error_ratio|log1p(c_j/max(h_j,1e−12))|
|8|log_observed_variation_ratio|log1p(std_t(raw_j)/max(s_j,1e−12))|
|9|residual_rank|stable descending b rank divided by max(d−1,1)|
|10|signature_rank|stable ascending a rank divided by max(d−1,1)|
|11|context_supported|original dominant phase exists in original noise groups; NOT Context low_support or U|

A decision-fit StandardScaler transforms these features. Linear/tree scores pass through sigmoid with logits clipped to [−40,40]. They are ranking scores, not calibrated posterior probabilities. The learner comparison uses balanced logistic regression (`C=1`, 2,000 iterations) and boosted trees (100 iterations, seven leaves, depth three, minimum leaf size 20, learning rate 0.1, L2 penalty 1, no early stopping). Both use seed 31017. Eligible learner selection maximizes beneficial risk-role count, then minimizes wrong selections and prefers linear on ties. Context retains the configuration's selected scorer.

A positive development label requires a valid candidate, the correct drift/exact-stuck target and lower clean-reference MAE. Other candidates and controls are negative. Clean references and labels are unavailable during assessment prediction. The seed-23 scorer and confidence rules transfer unchanged across checkpoints, while residual profiles are checkpoint-specific.

A relative proposal requires a unique finite top score, all candidates valid and a strict gap above `tau`. Top scores within NumPy `isclose(rtol=1e-6, atol=1e-12)` abstain. The gap is top score minus the largest alternative score. Risk calibration searches 20 gaps from 0 to 0.95, requiring nonempty selections, at most 5% incorrect/nonbeneficial selections and at most 5% unchanged-normal referrals. It maximizes beneficial count, then minimizes wrong count and prefers the larger gap. No feasible gap disables proposals. The three unchanged risk observations give a zero-of-three screen, not prospective 5% control.

Selection and Combined instead use the pooled boosted absolute scorer. Exactly one valid candidate must strictly exceed its threshold. Invalid alternatives are excluded rather than enforcing all-candidate validity. Reference and Selection use shared-history normalization, while Context and Combined use component history. The history ladder reports both separately fitted gaps and the fixed full-Context gap. Normalization still changes event admission in both comparisons.

## MS-05 — TTO and numerical validity

TTO initializes masked raw cells with the training mean and applies the model's scaler. Observed cells remain unchanged. Frozen model weights run in evaluation mode. Adam with learning rate 0.01 minimizes half the observed-cell reconstruction MSE plus 0.001 times half the squared consecutive differences at missing inputs. Denominators use count plus 1e-8. Per-sample missing-input gradients are normalized by L2 norm floored at 1e-8.

The returned estimate is the AE reconstruction at the final optimized input, rather than the optimized missing input itself. Only masked cells are copied into the corrected observation.

The limit is 5,000 cycles. Bootstrap uses the first 100 batch-mean losses. At cycle 100, the improvement tolerance becomes 0.01 times their mean. Plateau tolerance is 0.01 times that tolerance. Strict improvement resets the nonimprovement counter. Stop each sample after 100 consecutive nonimprovements or 200 plateau cycles, freezing converged samples. Return the final iterate without restoring a best-input checkpoint. The candidate-batch wall-time limit is 60 seconds.

A candidate is valid only when it converges without a failure reason. Nonfinite losses, gradients, values or final outputs, timeouts, interruption and maximum-cycle exhaustion remain terminal outcomes. Observed cells must be preserved and model state immutable. A failed alternative blocks relative selection. Numerical convergence does not establish reconstruction quality.

## MS-06 — Process flag and routing

For output-channel residual MSE, influence is the mean absolute input derivative including residual and network paths. Normalize each influence row over inputs. A normal q95 and scale `max(q95−median,1e-6*abs(q95),1e-12)` define nonnegative residual excess. Weight residual excess by normalized influence and sum across output channels. Separate sensitivity calibration gives excess and a strict normal-calibrated peak threshold.

The process flag `F` requires a peak in all five windows, at least two channels persistent in four windows, distinct declared component owners and a connected induced graph of those owners. Only declared system connections count. Paths through nonpeak components and hierarchy edges do not count. Disconnected or incomplete graphs can leave `F=0` despite distributed evidence. The supported-native panel contains zero flags across 25 observations.

Context eligibility `U` requires available, valid context under the trusted-source assumption. For relative selection, `K` combines `U`, all-candidate validity, a unique rank winner and strict gap exceedance. Then `P=A and K and not F`, `V=A and K and F`, and `C=A or F or not U or native_AE_alarm`. The native alarm uses any scoring window exceeding the model's native threshold. It changes concern, never proposal acceptance. Proposals and previews do not apply measurement replacement or plant actions.

## MS-07 — Source roles and boundaries

Normal-reference periods fit normalizers. Separate normal-calibration periods set event limits. Dictionary periods supply anchors and templates. Decision-fit records fit candidate scorers, and risk-calibration records select learners and gaps. Native development stays outside assessment. Source-role memberships can overlap between stages and are not independent recordings. Primary faults and paired controls use 132 source periods from sixteen files/batches. All controls bring this to 180 periods. The twelve primary SWaT periods share one normal recording.

The primary panel is a retrospective policy comparison. Temporal assessment uses distinct intervals of the same SWaT recording and does not establish source-independent transfer. Checkpoints, channels and neighboring periods share sources. The selected fault mixture is not deployment prevalence. Closed Parquet tables retain case IDs, source coordinates and available source identities. The fresh source-role planner and boundary rules are versioned in the package and configuration.

## MS-08 — Acquisition and fresh training

Complete recordings and archived model weights are not included. See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for provider links and [WORKFLOW.md](WORKFLOW.md) for fresh-training settings and source partitions.

The fresh recipe uses projection 64, one-layer encoder and decoder LSTMs of hidden size 128, residual MLP width 512 and bottleneck `max(1,min(16,floor(0.5*channels)))`. It uses training-only min–max scaling, length-60 windows and stride ten. Fresh-run manifests bind sources, scalers, models, profiles and terminal predictions for resume and evaluation. Those run-specific identifiers remain part of the implementation.

## MS-09 — Controls and quality accounting

Controls distinguish unchanged normal observations, response-supported native conditions, locally unverified native conditions, configured-no-response native conditions, witness drift, common-mode corruption and missing/misaligned/untrusted context. Qualification is retained separately from measured reconstruction quality. HAI native process faults are simulated. SWaT includes attacks. FourTank deviation IDs alone do not identify measured-effect onset.

The evaluator joins clean values, fault family/amplitude and true target only after prediction closure. Correct-beneficial means the corrupted sensor is selected and its MAE decreases. Strict harm means selected-channel MAE increases. Signed benefit is observed minus reconstructed selected-channel MAE divided by its training range. Abstention contributes zero where quality exists, wrong-channel damage stays negative and remaining injected-target error is reported separately. Native missing counterfactuals remain missing. Witness/common-mode improvement concerns the selected channel and does not establish complete process recovery.

## MS-10 — Reproduction

```bash
python -B scripts/reproduce.py --out /tmp/safeprocess-results
python -B scripts/reproduce.py --verify-only --out /tmp/safeprocess-check
```

The result guide maps E1–E6 and M1 to inputs, commands and output tables. These commands aggregate closed outcomes, replay numerical decision fixtures and render figures. They execute no training, TTO, profile fitting or policy selection. Use `safeprocess run` for new training and evaluation.
