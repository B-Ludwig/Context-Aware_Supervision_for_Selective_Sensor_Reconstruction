"""Deterministic, segment-aware fresh training and portable model bundles.

This recipe deliberately has a fresh identity. Its weights do not purport to
reproduce the archived checkpoints or their reported counts.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import LSTMProjMLPNet, architecture


RECIPE_ID = "safeprocess-fresh-projlstm-v1"
DEFAULTS = dict(input_proj_dim=64, hidden_size=128, latent_fraction=.5,
                latent_cap=16, n_layers=1, post_mlp_ratio=4, dropout=.1,
                post_mlp_dropout=0., batch_size=512, max_epochs=3000,
                lr=.001, patience=100, min_delta=.001, grad_clip=1.,
                window_length=60, stride=10, weight_decay=0.)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _torch_save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def resolve_device(device: str = "cpu") -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and explicitly requested CUDA are supported")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class TrainingScaler:
    minimum: np.ndarray
    ranges: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def divisor(self) -> np.ndarray:
        return np.where(self.ranges > 0., self.ranges, 1.)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((np.asarray(values, dtype=np.float64) - self.minimum)
                          / self.divisor, dtype=np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.divisor + self.minimum

    inverse_transform = inverse

    @classmethod
    def fit(cls, segments: list[np.ndarray]) -> "TrainingScaler":
        # Each source row contributes once, rather than once per overlapping window.
        count = sum(len(segment) for segment in segments)
        total = sum((segment.sum(0, dtype=np.float64) for segment in segments),
                    np.zeros(segments[0].shape[1], dtype=np.float64))
        mean = total / count
        var = sum((((segment - mean) ** 2).sum(0) for segment in segments),
                  np.zeros_like(mean)) / count
        minimum = np.minimum.reduce([segment.min(0) for segment in segments])
        maximum = np.maximum.reduce([segment.max(0) for segment in segments])
        return cls(minimum, maximum - minimum, mean, np.sqrt(var))

    def save(self, path: Path) -> None:
        np.savez(path, minimum=self.minimum, ranges=self.ranges,
                 mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "TrainingScaler":
        with np.load(path, allow_pickle=False) as values:
            return cls(*(values[key].copy() for key in ("minimum", "ranges", "mean", "std")))


class SegmentWindows(Dataset):
    """Sliding windows within individual partitions, never across boundaries."""

    def __init__(self, segments: list[np.ndarray], scaler: TrainingScaler,
                 length: int = 60, stride: int = 10):
        self.segments = [torch.from_numpy(scaler.transform(x)) for x in segments]
        self.length, self.stride = length, stride
        counts = [max(0, (len(x) - length) // stride + 1) for x in segments]
        self.ends = np.cumsum(counts).tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError("Each role needs at least one complete window")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        segment = bisect_right(self.ends, index)
        previous = self.ends[segment - 1] if segment else 0
        start = (index - previous) * self.stride
        return self.segments[segment][start:start + self.length]


def _validated_segments(segments, width, role):
    arrays = [np.asarray(segment, dtype=np.float64) for segment in segments]
    if not arrays:
        raise ValueError(f"{role} segments are empty")
    for value in arrays:
        if value.ndim != 2 or value.shape[1] != width or not len(value):
            raise ValueError(f"{role} segments must be nonempty [rows,{width}] arrays")
        if not np.isfinite(value).all():
            raise ValueError(f"{role} data must be finite")
    return arrays


def _binding(config, features, seed, roles):
    h = hashlib.sha256(json.dumps(dict(recipe=RECIPE_ID, config=config,
                                     features=features, seed=seed),
                                sort_keys=True).encode())
    for role, segments in roles.items():
        h.update(role.encode())
        for segment in segments:
            h.update(str(segment.shape).encode())
            h.update(np.asarray(segment, dtype="<f8").tobytes())
    return h.hexdigest()


def _rng_state(generator, device):
    result = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch=torch.get_rng_state(), loader=generator.get_state())
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state, generator, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _validation_loss(model, loader, device):
    model.eval()
    total = 0.
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            total += float((model(batch) - batch).square().sum())
            count += batch.numel()
    return total / count


def train_bundle(train_segments: list[np.ndarray], validation_segments: list[np.ndarray],
                 calibration_segments: list[np.ndarray], *, feature_names: list[str],
                 config: dict, seed: int, out: Path, device="cpu", resume=False) -> dict:
    """Train and save a bundle, resuming only an identical binding.

    Administrative ``stop_after_epochs`` and ``max_seconds`` permit bounded
    checks. An interrupted epoch is replayed from its start on resume. They do
    not change the scientific recipe or qualify an incomplete bundle for use.
    """
    device = resolve_device(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be nonempty and unique")
    scientific = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    if "learning_rate" in config and "lr" not in config:
        scientific["lr"] = config["learning_rate"]
    if scientific["weight_decay"] != 0:
        raise ValueError("The fresh recipe has no weight decay")
    if int(scientific["max_epochs"]) < 1 or int(scientific["batch_size"]) < 1:
        raise ValueError("Epoch and batch sizes must be positive")
    if not 0 <= float(scientific["min_delta"]) < 1:
        raise ValueError("Relative min_delta must be in [0,1)")
    if int(scientific["patience"]) < 1:
        raise ValueError("Training patience must be positive")
    width = len(feature_names)
    roles = {name: _validated_segments(values, width, name) for name, values in (
        ("train", train_segments), ("validation", validation_segments),
        ("calibration", calibration_segments))}
    binding = _binding(scientific, feature_names, seed, roles)
    if any((out / name).exists() for name in ("bundle.json", "last.pt", "best.pt", "scaler.npz")) and not resume:
        raise FileExistsError("Bundle already exists; use resume with matching inputs")
    seed_all(seed, device)
    scaler = TrainingScaler.fit(roles["train"])
    if resume and (out / "bundle.json").exists():
        existing = json.loads((out / "bundle.json").read_text())
        if existing["binding"] != binding:
            raise ValueError("Resume inputs, source roles, feature order or recipe changed")
        if existing["completed"]:
            load_bundle(out, device=str(device))
            return existing
    resume_state = None
    if resume:
        if not (out / "last.pt").exists():
            raise FileNotFoundError("Resume requires the retained local last.pt checkpoint")
        # Check the binding before writing even a scaler into the destination.
        resume_state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if resume_state["binding"] != binding:
            raise ValueError("Resume checkpoint binding changed")
    scaler.save(out / "scaler.npz")
    datasets = {role: SegmentWindows(segments, scaler, int(scientific["window_length"]),
                                     int(scientific["stride"]))
                for role, segments in roles.items()}
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(datasets["train"], batch_size=int(scientific["batch_size"]),
                        shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(datasets["validation"], batch_size=int(scientific["batch_size"]),
                            shuffle=False, num_workers=0)
    arch = architecture(width, scientific)
    model = LSTMProjMLPNet(**arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(scientific["lr"]), weight_decay=0.)
    history, epoch, stale = [], 0, 0
    best_loss = patience_reference = float("inf")
    best_epoch = -1
    if resume_state is not None:
        # Resume files are local training state, not untrusted downloaded weights.
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, epoch, stale = state["history"], state["epoch"], state["stale"]
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        patience_reference = state["patience_reference"]
        _restore_rng(state["rng"], generator, device)

    def checkpoint():
        _torch_save(out / "last.pt", dict(binding=binding, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), history=history, epoch=epoch,
                    stale=stale, best_loss=best_loss, best_epoch=best_epoch,
                    patience_reference=patience_reference, rng=_rng_state(generator, device)))

    if not (out / "last.pt").exists():
        checkpoint()
    started = time.monotonic()
    invocation_epochs = 0
    interrupted = False
    try:
        while epoch < int(scientific["max_epochs"]) and stale < int(scientific["patience"]):
            if config.get("stop_after_epochs") is not None and invocation_epochs >= int(config["stop_after_epochs"]):
                interrupted = True
                break
            if config.get("max_seconds") is not None and time.monotonic() - started >= float(config["max_seconds"]):
                interrupted = True
                break
            model.train()
            train_total, train_cells = 0., 0
            for batch in loader:
                if config.get("max_seconds") is not None and time.monotonic() - started >= float(config["max_seconds"]):
                    interrupted = True
                    break
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(batch), batch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(scientific["grad_clip"]),
                                         error_if_nonfinite=True)
                optimizer.step()
                train_total += float(loss.detach()) * batch.numel()
                train_cells += batch.numel()
            if interrupted:
                break
            val_loss = _validation_loss(model, val_loader, device)
            if not np.isfinite(val_loss):
                raise RuntimeError("Nonfinite validation loss")
            improvement = (val_loss < patience_reference
                           and val_loss <= patience_reference * (1. - float(scientific["min_delta"])))
            if improvement:
                patience_reference, stale = val_loss, 0
            else:
                stale += 1
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                _torch_save(out / "best.pt", {key: value.detach().cpu()
                                             for key, value in model.state_dict().items()})
            history.append(dict(epoch=epoch, train_loss=train_total / train_cells,
                                validation_loss=val_loss, patience_reset=bool(improvement)))
            epoch += 1
            invocation_epochs += 1
            checkpoint()
    except KeyboardInterrupt:
        interrupted = True
    completed = not interrupted and (epoch >= int(scientific["max_epochs"])
                                     or stale >= int(scientific["patience"]))
    native_threshold = None
    if completed:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
        model.eval()
        cal_loader = DataLoader(datasets["calibration"], batch_size=int(scientific["batch_size"]),
                                shuffle=False, num_workers=0)
        calibration_scores = []
        with torch.no_grad():
            for batch in cal_loader:
                batch = batch.to(device)
                calibration_scores.extend((model(batch) - batch).square().mean((1, 2)).cpu().tolist())
        native_threshold = float(np.quantile(calibration_scores, .95, method="higher"))
    metadata = dict(schema="safeprocess.model-bundle.v1", recipe=RECIPE_ID,
                    binding=binding, completed=completed, status="complete" if completed else "interrupted",
                    seed=seed, feature_names=feature_names, architecture=arch,
                    training_config=scientific, epoch=epoch, best_epoch=best_epoch,
                    best_validation_loss=best_loss if np.isfinite(best_loss) else None,
                    history=history, native_threshold=native_threshold,
                    native_threshold_definition="higher empirical 95th percentile of normal calibration window mean scaled MSE",
                    role_windows={name: len(value) for name, value in datasets.items()},
                    scaler_file="scaler.npz", weights_file="best.pt",
                    scaler_sha256=sha256_file(out / "scaler.npz"),
                    weights_sha256=sha256_file(out / "best.pt") if (out / "best.pt").exists() else None)
    _json(out / "bundle.json", metadata)
    return metadata


@dataclass
class ModelBundle:
    model: nn.Module
    device: torch.device
    feature_names: list[str]
    scaler: TrainingScaler
    native_threshold: float
    metadata: dict

    @property
    def training_mean(self):
        return self.scaler.mean

    @property
    def training_ranges(self):
        return self.scaler.ranges

    @property
    def training_std(self):
        return self.scaler.std

    def forward_scaled(self, values: np.ndarray) -> np.ndarray:
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.as_tensor(np.asarray(values), dtype=torch.float32,
                                                  device=self.device)).cpu().numpy()
        finally:
            for module, training in modes:
                module.training = training


def load_bundle(path: Path, device="cpu") -> ModelBundle:
    path = Path(path)
    if path.is_dir():
        path = path / "bundle.json"
    metadata = json.loads(path.read_text())
    if metadata.get("schema") != "safeprocess.model-bundle.v1" or not metadata.get("completed"):
        raise ValueError("Only a completed supported model bundle can be loaded")
    for name, hash_key in (("scaler_file", "scaler_sha256"), ("weights_file", "weights_sha256")):
        filename = Path(metadata[name])
        if filename.name != str(filename):
            raise ValueError("Bundle files must be local filenames")
        if sha256_file(path.parent / filename) != metadata[hash_key]:
            raise ValueError(f"Bundle integrity failure: {filename}")
    device = resolve_device(device)
    model = LSTMProjMLPNet(**metadata["architecture"]).to(device)
    model.load_state_dict(torch.load(path.parent / metadata["weights_file"],
                                    map_location=device, weights_only=True))
    model.eval()
    return ModelBundle(model, device, metadata["feature_names"],
                       TrainingScaler.load(path.parent / metadata["scaler_file"]),
                       float(metadata["native_threshold"]), metadata)
