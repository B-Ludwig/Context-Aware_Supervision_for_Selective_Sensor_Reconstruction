"""Shared observable evidence and masked input-gradient reconstruction.

No evaluator labels or clean reference enter this module. Candidate validity
requires convergence, not merely exhaustion of an optimization budget.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
import torch


WINDOW_STARTS = (200, 210, 220, 230, 240)
TTO_DEFAULTS = dict(optimizer="adam", learning_rate=.01, max_cycles=5000,
                    patience=100, max_consecutive_cycles=200, init_window=100,
                    smooth_lambda=.001, max_seconds=60., projection_sigma=None,
                    tto_method_id="input-gradient-tto-prospective-v2",
                    tto_stopping_rule="finite-bootstrap-consecutive-patience-v2")


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@contextmanager
def frozen_model(model):
    """Freeze parameters and mode, then restore the caller's complete state."""
    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    before = model_fingerprint(model)
    try:
        model.eval()
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        # Native LSTM supports input gradients in eval mode on either device.
        with torch.backends.cudnn.flags(enabled=False):
            yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training
        if model_fingerprint(model) != before:
            raise RuntimeError("Frozen numerical inference modified model state")


@dataclass
class TTOResult:
    values: torch.Tensor
    optimized_inputs: torch.Tensor
    reconstruction: torch.Tensor
    cycles: int
    cycles_per_sample: tuple[int, ...]
    converged_per_sample: tuple[bool, ...]
    final_loss_per_sample: tuple[float, ...]
    failure_reasons: tuple[str | None, ...]
    seconds: float
    min_delta: float


def _per_sample_loss(reconstruction, candidate, observed_values, observed_mask,
                     smooth_lambda):
    weight = observed_mask.to(reconstruction.dtype)
    safe_targets = torch.where(observed_mask, observed_values, reconstruction)
    losses = .5 * ((reconstruction - safe_targets).square() * weight).sum((1, 2)) / (weight.sum((1, 2)) + 1e-8)
    if smooth_lambda > 0 and candidate.shape[1] > 1:
        missing = ~observed_mask
        touching = (missing[:, 1:] & missing[:, :-1]).to(candidate.dtype)
        differences = candidate[:, 1:] - candidate[:, :-1]
        losses += smooth_lambda * .5 * (differences.square() * touching).sum((1, 2)) / (touching.sum((1, 2)) + 1e-8)
    return losses


def _settings(config):
    settings = {**TTO_DEFAULTS, **config}
    if str(settings["optimizer"]).lower() != "adam":
        raise ValueError("The frozen TTO rule uses Adam")
    if settings["projection_sigma"] is not None:
        raise ValueError("The frozen paper TTO rule has no projection")
    if settings["tto_method_id"] != TTO_DEFAULTS["tto_method_id"] or settings["tto_stopping_rule"] != TTO_DEFAULTS["tto_stopping_rule"]:
        raise ValueError("Only the explicitly identified prospective TTO rule is supported")
    if config.get("disable_early_stopping", False):
        raise ValueError("Budget exhaustion cannot replace convergence")
    if int(settings["max_cycles"]) < 1 or int(settings["init_window"]) < 0:
        raise ValueError("Invalid TTO cycle or bootstrap count")
    if min(int(settings["patience"]), int(settings["max_consecutive_cycles"])) < 1:
        raise ValueError("TTO patience counts must be positive")
    for key in ("learning_rate", "smooth_lambda", "max_seconds", "min_delta", "absolute_monotonic_deadline"):
        value = settings.get(key)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"Invalid finite nonnegative TTO setting: {key}")
    return settings


def input_optimization(model, observed_values: torch.Tensor,
                       observed_mask: torch.Tensor, fill_value: torch.Tensor,
                       config: dict | None = None) -> TTOResult:
    """Optimize missing inputs, returning the decoder estimate at the last iterate.

    Hidden input cells may contain arbitrary sentinels or NaNs. They are never
    read as targets or initialization. The training mean is the only fill.
    """
    settings = _settings(config or {})
    if observed_values.ndim != 3 or observed_values.shape != observed_mask.shape:
        raise ValueError("TTO requires matching [batch,time,feature] inputs and masks")
    if not observed_values.shape[0]:
        raise ValueError("TTO requires at least one candidate")
    observed_mask = observed_mask.bool()
    if not bool(observed_mask.flatten(1).any(1).all()):
        raise ValueError("Every candidate needs observed evidence")
    if not bool(torch.isfinite(observed_values[observed_mask]).all()):
        raise ValueError("Observed evidence must be finite")
    fill = torch.as_tensor(fill_value, device=observed_values.device,
                           dtype=observed_values.dtype).reshape(1, 1, -1)
    if fill.shape[-1] != observed_values.shape[-1] or not bool(torch.isfinite(fill).all()):
        raise ValueError("A finite training mean is required for each channel")
    with frozen_model(model):
        return _optimize(model, observed_values, observed_mask, fill, settings)


def _optimize(model, observed_values, observed_mask, fill, settings):
    baseline = torch.where(observed_mask, observed_values, fill.expand_as(observed_values))
    optimized = baseline.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized], lr=float(settings["learning_rate"]))
    batch = observed_values.shape[0]
    best = torch.full((batch,), float("inf"), device=observed_values.device)
    previous = best.clone()
    no_improve = torch.zeros(batch, dtype=torch.long, device=observed_values.device)
    stable = torch.zeros_like(no_improve)
    converged = torch.zeros(batch, dtype=torch.bool, device=observed_values.device)
    converge_iter = torch.full_like(no_improve, -1)
    terminal_reasons = [None] * batch
    initial_losses = []
    min_delta = float(settings.get("min_delta") or 0.)
    delta_threshold = 0.
    init_window = int(settings["init_window"])
    cycles = 0
    timed_out = False
    global_failure = None
    started = time.monotonic()
    for cycle in range(int(settings["max_cycles"])):
        now = time.monotonic()
        if ((settings["max_seconds"] is not None and now - started >= float(settings["max_seconds"]))
                or (settings.get("absolute_monotonic_deadline") is not None
                    and now >= float(settings["absolute_monotonic_deadline"]))):
            timed_out = True
            break
        optimizer.zero_grad(set_to_none=True)
        active = ~converged
        frozen = optimized[converged].detach().clone() if bool(converged.any()) else None
        candidate = torch.where(observed_mask, observed_values, optimized)
        reconstruction = model(candidate)
        per_sample = _per_sample_loss(reconstruction, candidate, observed_values,
                                      observed_mask, float(settings["smooth_lambda"]))
        if not bool(torch.isfinite(per_sample).all()):
            global_failure = "non_finite_loss"
            for index in range(batch):
                if bool(active[index]):
                    terminal_reasons[index] = (global_failure if not bool(torch.isfinite(per_sample[index]))
                                               else "batch_interrupted_non_finite_loss")
            break
        (per_sample * active.to(per_sample.dtype)).sum().backward()
        if optimized.grad is None or not bool(torch.isfinite(optimized.grad).all()):
            global_failure = "non_finite_gradient"
            finite = (torch.zeros(batch, dtype=torch.bool, device=optimized.device)
                      if optimized.grad is None else torch.isfinite(optimized.grad).flatten(1).all(1))
            for index in range(batch):
                if bool(active[index]):
                    terminal_reasons[index] = (global_failure if not bool(finite[index])
                                               else "batch_interrupted_non_finite_gradient")
            break
        with torch.no_grad():
            optimized.grad.mul_((~observed_mask).to(optimized.grad.dtype))
            optimized.grad.mul_(active.reshape(-1, 1, 1))
            norms = optimized.grad.flatten(1).norm(dim=1).clamp_min(1e-8)
            optimized.grad.div_(norms.reshape(-1, 1, 1))
        optimizer.step()
        with torch.no_grad():
            optimized[observed_mask] = observed_values[observed_mask]
            if frozen is not None:
                optimized[converged] = frozen
        current = per_sample.detach()
        if cycle < init_window:
            initial_losses.append(float(current.mean()))
        if cycle == init_window:
            best = current.clone()
            initial = float(np.mean(initial_losses)) if initial_losses else float(current.mean())
            min_delta = .01 * initial if settings.get("min_delta") is None else float(settings["min_delta"])
            delta_threshold = .01 * min_delta
        if cycle > init_window:
            improved = active & (current < best - min_delta)
            best[improved] = current[improved]
            no_improve[improved] = 0
            no_improve[active & ~improved] += 1
            moving = active & ((previous - current).abs() > delta_threshold)
            stable[moving] = 0
            stable[active & ~moving] += 1
        previous = current.clone()
        if cycle >= init_window:
            newly = active & ((no_improve >= int(settings["patience"]))
                              | (stable >= int(settings["max_consecutive_cycles"])))
            converge_iter[newly] = cycle + 1
            converged |= newly
        cycles = cycle + 1
        if bool(converged.all()):
            break
    candidate = torch.where(observed_mask, observed_values, optimized.detach())
    with torch.no_grad():
        reconstruction = model(candidate)
        losses = _per_sample_loss(reconstruction, candidate, observed_values,
                                  observed_mask, float(settings["smooth_lambda"]))
        values = torch.where(observed_mask, observed_values, reconstruction)
    for reason, invalid in (("non_finite_optimized_input", ~torch.isfinite(candidate).flatten(1).all(1)),
                             ("non_finite_output", ~torch.isfinite(reconstruction).flatten(1).all(1)),
                             ("non_finite_final_loss", ~torch.isfinite(losses))):
        for index in invalid.nonzero(as_tuple=False).flatten().tolist():
            if terminal_reasons[index] is None:
                terminal_reasons[index] = reason
            converged[index] = False
    reasons = tuple(terminal_reasons[i] if terminal_reasons[i] is not None
                    else None if bool(converged[i])
                    else "wall_time_limit" if timed_out
                    else f"batch_interrupted_{global_failure}" if global_failure
                    else "max_cycles" for i in range(batch))
    return TTOResult(values.detach(), candidate.detach(), reconstruction.detach(), cycles,
                     tuple(int(value) if int(value) >= 0 else cycles for value in converge_iter),
                     tuple(bool(value) for value in converged),
                     tuple(float(value) for value in losses), reasons,
                     time.monotonic() - started, min_delta)


def residual_evidence(bundle, windows: np.ndarray):
    """Residuals and absolute input Jacobians, including the direct residual path."""
    scaled = bundle.scaler.transform(windows)
    x = torch.as_tensor(scaled, device=bundle.device).detach().requires_grad_(True)
    with frozen_model(bundle.model):
        reconstruction = bundle.model(x)
        signed = x - reconstruction
        losses = signed.square().mean(1)
        influence = []
        for channel in range(x.shape[-1]):
            gradient = torch.autograd.grad(losses[:, channel].sum(), x,
                                            retain_graph=channel + 1 < x.shape[-1])[0]
            influence.append(gradient.detach().abs().mean(1).cpu().numpy())
        return (signed.detach().cpu().numpy(), losses.detach().cpu().numpy(),
                np.stack(influence, axis=1))


def compute_evidence(bundle, raw300: np.ndarray, *, tto_config: dict,
                     candidates=True):
    """Build one shared numerical payload for every policy on an observation."""
    raw = np.asarray(raw300, dtype=np.float64)
    width = len(bundle.feature_names)
    if raw.shape != (300, width) or not np.isfinite(raw).all():
        raise ValueError(f"Expected a finite [300,{width}] observation")
    if width < 2 and candidates:
        raise ValueError("Target masking requires at least one other observed channel")
    fingerprint = model_fingerprint(bundle.model)
    windows = np.stack([raw[start:start + 60] for start in WINDOW_STARTS])
    signed, residuals, influence = residual_evidence(bundle, windows)
    raw_candidates = np.repeat(windows[-1][None], width, axis=0)
    detail = dict(schema="safeprocess.evidence.v1", window_starts=list(WINDOW_STARTS),
                  feature_names=list(bundle.feature_names), model_fingerprint=fingerprint,
                  tto_settings=_settings(tto_config), candidates_requested=bool(candidates))
    if candidates:
        observed = np.ones_like(raw_candidates, dtype=bool)
        for channel in range(width):
            observed[channel, :, channel] = False
        # Strip the proposed channel before scaling, preventing accidental use
        # of its readings even when they are extreme or nonfinite sentinels.
        hidden = np.where(observed, raw_candidates, np.nan)
        scaled = torch.as_tensor(bundle.scaler.transform(hidden), device=bundle.device)
        mask = torch.as_tensor(observed, device=bundle.device)
        fill = torch.as_tensor(bundle.scaler.transform(bundle.training_mean), device=bundle.device)
        result = input_optimization(bundle.model, scaled, mask, fill, tto_config)
        inverse = bundle.scaler.inverse(result.values.cpu().numpy())
        corrected = raw_candidates.copy()
        corrected[~observed] = inverse[~observed]
        valid = np.array([converged and reason is None
                          for converged, reason in zip(result.converged_per_sample,
                                                       result.failure_reasons)], dtype=bool)
        detail.update(cycles=result.cycles, seconds=result.seconds,
                       bootstrap_min_delta=result.min_delta,
                       candidates=[dict(channel=channel, feature=bundle.feature_names[channel],
                                        converged=result.converged_per_sample[channel],
                                        cycles=result.cycles_per_sample[channel],
                                        loss=(result.final_loss_per_sample[channel]
                                              if math.isfinite(result.final_loss_per_sample[channel]) else None),
                                        failure_reason=result.failure_reasons[channel])
                                   for channel in range(width)])
        corrected_scaled = bundle.scaler.transform(corrected)
        finite = np.isfinite(corrected_scaled).all((1, 2))
        after = np.full((width, width), np.nan)
        if finite.any():
            output = bundle.forward_scaled(corrected_scaled[finite])
            after[finite] = ((corrected_scaled[finite] - output) ** 2).mean(1)
    else:
        corrected = raw_candidates.copy()
        valid = np.zeros(width, dtype=bool)
        after = np.repeat(residuals[-1][None], width, axis=0)
        detail.update(cycles=0, seconds=0., candidates=[
            dict(channel=channel, feature=bundle.feature_names[channel], converged=False,
                 cycles=0, loss=None, failure_reason="not_requested") for channel in range(width)])
    keep = ~np.eye(width, dtype=bool)
    before_other = np.where(keep, residuals[-1][None], 0.).sum(1)
    after_other = np.where(keep, after, 0.).sum(1)
    fractions = (before_other - after_other) / np.maximum(before_other, 1e-12)
    arrays = dict(raw=raw.copy(), signed=signed, residuals=residuals,
                   influence=influence, corrected=corrected, valid=valid,
                   after=after, fractions=fractions)
    if model_fingerprint(bundle.model) != fingerprint:
        raise RuntimeError("Evidence generation modified model state")
    return canonicalize_candidates(arrays, detail)


def canonicalize_candidates(arrays, detail):
    """Retain nonfinite candidate failures using observable finite placeholders.

    Finite candidates, including finite nonconverged candidates, are unchanged.
    A placeholder is never an eligible reconstruction or evaluator estimate.
    """
    for name in ("raw", "signed", "residuals", "influence"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError("Nonfinite base observation evidence: " + name)
    bad=(~np.isfinite(arrays["corrected"]).all((1,2)) |
         ~np.isfinite(arrays["after"]).all(1) |
         ~np.isfinite(arrays["fractions"]))
    detail["invalid_placeholder_schema"]="nonfinite-candidate-observed-v1"
    detail["placeholder_channels"]=np.flatnonzero(bad).tolist()
    for j in np.flatnonzero(bad):
        arrays["valid"][j]=False
        arrays["corrected"][j]=arrays["raw"][-60:]
        arrays["after"][j]=arrays["residuals"][-1]
        arrays["fractions"][j]=0.
        candidate=detail["candidates"][int(j)]
        candidate["converged"]=False
        candidate["failure_reason"]=candidate.get("failure_reason") or "nonfinite_candidate_evidence"
        candidate["observable_placeholder"]=True
    return arrays,detail
