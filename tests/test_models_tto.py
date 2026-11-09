"""Bounded CPU checks for fresh training and observable reconstruction kernels."""

import json

import numpy as np
import pytest
import torch

from safeprocess.models import build_model
from safeprocess.numerics import (compute_evidence, input_optimization,
                                   model_fingerprint, residual_evidence)
from safeprocess.training import (ModelBundle, TrainingScaler, load_bundle,
                                  train_bundle)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def data():
    time = np.arange(180) / 20
    train = np.stack((np.sin(time), np.cos(time), .5 * np.sin(time) + .2), axis=1)
    return [train], [train + .03], [train + .01]


def config():
    return dict(input_proj_dim=4, hidden_size=4, latent_cap=2, max_epochs=3,
                batch_size=8, patience=10, post_mlp_ratio=2)


@pytest.fixture
def bundle(tmp_path):
    train, validation, calibration = data()
    train_bundle(train, validation, calibration, feature_names=["a", "b", "c"],
                 config=config(), seed=23, out=tmp_path / "model")
    return load_bundle(tmp_path / "model")


def test_architecture_and_training_only_scaler(tmp_path):
    model = build_model(11, {})
    assert model.bottleneck.out_features == 5
    assert model.encoder.num_layers == model.decoder.num_layers == 1
    assert model.encoder.dropout == model.decoder.dropout == 0
    assert model.post_mlp.drop.p == 0
    train, validation, calibration = data()
    train_bundle(train, [validation[0] + 50], [calibration[0] + 100],
                 feature_names=["a", "b", "c"], config=config(), seed=23,
                 out=tmp_path / "model")
    loaded = load_bundle(tmp_path / "model")
    assert np.array_equal(loaded.scaler.minimum, train[0].min(0))
    assert np.array_equal(loaded.training_ranges, np.ptp(train[0], axis=0))
    assert np.allclose(loaded.training_mean, train[0].mean(0), atol=1e-15)
    assert loaded.native_threshold > 100


def test_deterministic_training_and_exact_resume(tmp_path):
    train, validation, calibration = data()
    arguments = dict(feature_names=["a", "b", "c"], seed=23)
    full = train_bundle(train, validation, calibration, config=config(),
                        out=tmp_path / "full", **arguments)
    interrupted = train_bundle(train, validation, calibration,
                               config={**config(), "stop_after_epochs": 1},
                               out=tmp_path / "resume", **arguments)
    assert interrupted["status"] == "interrupted"
    with pytest.raises(ValueError, match="completed"):
        load_bundle(tmp_path / "resume")
    resumed = train_bundle(train, validation, calibration, config=config(),
                           out=tmp_path / "resume", resume=True, **arguments)
    repeated = train_bundle(train, validation, calibration, config=config(),
                            out=tmp_path / "repeat", **arguments)
    assert full["history"] == resumed["history"] == repeated["history"]
    assert full["native_threshold"] == resumed["native_threshold"]
    reference = load_bundle(tmp_path / "full")
    assert model_fingerprint(reference.model) == model_fingerprint(load_bundle(tmp_path / "resume").model)
    assert model_fingerprint(reference.model) == model_fingerprint(load_bundle(tmp_path / "repeat").model)
    with pytest.raises(ValueError, match="changed"):
        train_bundle(train, validation, calibration, config=config(),
                     out=tmp_path / "resume", resume=True,
                     feature_names=["b", "a", "c"], seed=23)


def test_reload_and_integrity(bundle, tmp_path):
    original = bundle.forward_scaled(np.ones((2, 60, 3), np.float32))
    assert np.array_equal(original, bundle.forward_scaled(np.ones((2, 60, 3), np.float32)))
    model_dir = tmp_path / "model"
    with (model_dir / "best.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        load_bundle(model_dir)


def test_target_mask_invariance_decoder_estimate_and_frozen_state(bundle):
    observed = np.arange(2 * 60 * 3).reshape(2, 60, 3) / 300
    mask = np.ones_like(observed, dtype=bool)
    mask[0, :, 0] = mask[1, :, 1] = False
    fill = torch.tensor(bundle.scaler.transform(bundle.training_mean))
    options = dict(max_cycles=12, init_window=1, patience=2,
                   max_consecutive_cycles=2, min_delta=100.)
    before = model_fingerprint(bundle.model)
    bundle.model.train()
    a = torch.tensor(np.where(mask, observed, np.nan), dtype=torch.float32)
    b = torch.tensor(np.where(mask, observed, 987654.), dtype=torch.float32)
    left = input_optimization(bundle.model, a, torch.tensor(mask), fill, options)
    right = input_optimization(bundle.model, b, torch.tensor(mask), fill, options)
    assert torch.equal(left.values, right.values)
    assert left.failure_reasons == right.failure_reasons == (None, None)
    assert torch.equal(left.values[torch.tensor(mask)], a[torch.tensor(mask)])
    assert torch.equal(left.values[~torch.tensor(mask)], left.reconstruction[~torch.tensor(mask)])
    assert not torch.equal(left.values[~torch.tensor(mask)], left.optimized_inputs[~torch.tensor(mask)])
    assert model_fingerprint(bundle.model) == before
    assert bundle.model.training
    assert all(parameter.requires_grad for parameter in bundle.model.parameters())


def test_nonconvergence_and_timeout_are_retained(bundle):
    values = torch.zeros(1, 60, 3)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 0] = False
    fill = torch.zeros(3)
    exhausted = input_optimization(bundle.model, values, mask, fill, dict(max_cycles=1))
    assert exhausted.failure_reasons == ("max_cycles",)
    assert exhausted.converged_per_sample == (False,)
    timed_out = input_optimization(bundle.model, values, mask, fill, dict(max_seconds=0))
    assert timed_out.failure_reasons == ("wall_time_limit",)
    assert timed_out.cycles == 0


def test_payload_shapes_observed_cells_and_direct_jacobian(bundle):
    time = np.arange(300) / 20
    raw = np.stack((np.sin(time), np.cos(time), .5 * np.sin(time) + .2), 1)
    before = model_fingerprint(bundle.model)
    arrays, detail = compute_evidence(bundle, raw, tto_config=dict(
        max_cycles=12, init_window=1, patience=2,
        max_consecutive_cycles=2, min_delta=100.))
    assert set(arrays) == {"raw", "signed", "residuals", "influence", "corrected", "valid", "after", "fractions"}
    assert arrays["signed"].shape == (5, 60, 3)
    assert arrays["influence"].shape == (5, 3, 3)
    assert arrays["after"].shape == (3, 3)
    assert arrays["valid"].all()
    for target in range(3):
        keep = np.arange(3) != target
        assert np.array_equal(arrays["corrected"][target][:, keep], raw[-60:, keep])
    assert len(detail["candidates"]) == 3
    assert before == model_fingerprint(bundle.model)
    no_candidates, details = compute_evidence(bundle, raw, tto_config={}, candidates=False)
    assert not no_candidates["valid"].any()
    assert all(item["failure_reason"] == "not_requested" for item in details["candidates"])
    # A zero-output AE exposes the direct d(x^2)/dx path in the loss Jacobian.
    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    zero = ModelBundle(Zero(), torch.device("cpu"), bundle.feature_names,
                       bundle.scaler, 0., {})
    windows = np.repeat(raw[-60:][None], 5, 0)
    _, _, influence = residual_evidence(zero, windows)
    scaled = zero.scaler.transform(windows)
    expected = np.zeros_like(influence)
    for channel in range(3):
        expected[:, channel, channel] = np.abs(2 * scaled[:, :, channel] / 60).mean(1)
    assert np.allclose(influence, expected, atol=1e-8)
    # Central finite difference checks one real-network Jacobian entry.
    x = torch.tensor(bundle.scaler.transform(windows[:1]), requires_grad=True)
    loss = (x - bundle.model(x)).square().mean(1)[0, 0]
    derivative = torch.autograd.grad(loss, x)[0][0, 20, 1].item()
    plus, minus = x.detach().clone(), x.detach().clone()
    plus[0, 20, 1] += .001
    minus[0, 20, 1] -= .001
    with torch.no_grad():
        a = (plus - bundle.model(plus)).square().mean(1)[0, 0]
        b = (minus - bundle.model(minus)).square().mean(1)[0, 0]
    finite_difference = ((a - b) / .002).item()
    assert np.isclose(derivative, finite_difference, atol=2e-5, rtol=.05)


def test_nonfinite_invalid_candidate_has_observable_placeholder_only():
    from safeprocess.numerics import canonicalize_candidates
    raw=np.arange(600,dtype=float).reshape(300,2)
    corrected=np.repeat(raw[-60:][None],2,axis=0)
    corrected[1,:,1]=np.nan
    arrays=dict(raw=raw,signed=np.zeros((5,60,2)),residuals=np.ones((5,2)),
      influence=np.ones((5,2,2)),corrected=corrected,after=np.array([[2.,3.],[np.nan,np.nan]]),
      fractions=np.array([.3,np.nan]),valid=np.array([True,False]))
    detail=dict(candidates=[dict(failure_reason=None),dict(failure_reason="non_finite_output")])
    finite_first=corrected[0].copy()
    result,note=canonicalize_candidates(arrays,detail)
    assert result["valid"].tolist()==[True,False]
    np.testing.assert_array_equal(result["corrected"][0],finite_first)
    np.testing.assert_array_equal(result["corrected"][1],raw[-60:])
    np.testing.assert_array_equal(result["after"][1],np.ones(2))
    assert result["fractions"].tolist()==[.3,0.]
    assert note["placeholder_channels"]==[1]
    assert note["candidates"][1]["failure_reason"]=="non_finite_output"
