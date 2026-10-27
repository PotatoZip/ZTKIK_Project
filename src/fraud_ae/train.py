"""Phase 2 – training loop with early stopping on validation-legit MSE (ARCHITECTURE.md §3).

The autoencoder sees **legitimate rows only**. Model selection uses the MSE on the legitimate
part of the validation split, so no fraud label influences training or checkpoint choice.

Run-directory contract (``runs/<name>/``)::

    config.toml          verbatim snapshot of the config file the run was started from
    model.pt             best checkpoint: state_dict + input_dim + model config + best epoch
    history.csv          epoch, train_mse, val_legit_mse, lr, seconds
    train_summary.json   epochs run, best epoch / val MSE, parameter count, wall time, seed
    figures/training_curves.png
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config, ModelConfig
from .model import Autoencoder, build_model, count_parameters

CONFIG_FILE = "config.toml"
MODEL_FILE = "model.pt"
HISTORY_FILE = "history.csv"
SUMMARY_FILE = "train_summary.json"
HISTORY_COLUMNS = ("epoch", "train_mse", "val_legit_mse", "lr", "seconds")


def seed_everything(seed: int, num_threads: int | None = None) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` and enable deterministic algorithms (CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.use_deterministic_algorithms(True)
    if num_threads:
        torch.set_num_threads(num_threads)


class EarlyStopping:
    """Stop when the monitored value has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value`` for ``epoch``; return ``True`` if it is a new best."""
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@torch.no_grad()
def mse_per_sample(model: nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Per-row mean squared reconstruction error in eval mode; shape ``(n,)``."""
    was_training = model.training
    model.eval()
    out = [((model(xb) - xb) ** 2).mean(dim=1) for xb in X.split(batch_size)]
    model.train(was_training)
    return torch.cat(out) if out else torch.empty(0)


def _snapshot_config(cfg: Config, run_dir: Path) -> None:
    """Write ``config.toml``: a verbatim copy of the source file, or – when CLI overrides such as
    ``--epochs`` / ``--seed`` changed it – the *effective* config written by hand (stdlib has
    no TOML writer) with a header naming the source file."""
    from .config import load_config

    dst = run_dir / CONFIG_FILE
    src = Path(cfg.source_path) if cfg.source_path else None
    if src is not None and src.exists() and load_config(src) == cfg:
        shutil.copyfile(src, dst)
        return
    lines = [f"# effective config (CLI overrides applied) derived from {src or '<in-memory>'}", ""]
    for section, values in cfg.to_dict().items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    dst.write_text("\n".join(lines))


def train(cfg: Config, X_train: np.ndarray, X_val_legit: np.ndarray, run_dir: Path) -> Autoencoder:
    """Train on legitimate rows only; keep the checkpoint with the lowest val-legit MSE.

    Returns the model loaded with the best weights. Everything is deterministic given
    ``cfg.train.seed`` (manual seeded batching, deterministic torch algorithms, CPU).
    """
    tc = cfg.train
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(tc.seed, tc.num_threads)
    _snapshot_config(cfg, run_dir)

    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    Xva = torch.as_tensor(np.asarray(X_val_legit), dtype=torch.float32)
    if Xtr.ndim != 2 or Xtr.shape[0] == 0:
        raise ValueError("X_train must be a non-empty 2-D array")
    if Xva.ndim != 2 or Xva.shape[1] != Xtr.shape[1]:
        raise ValueError("X_val_legit must have the same number of columns as X_train")
    n, input_dim = Xtr.shape

    model = build_model(input_dim, cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, tc.patience // 2)
    )
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(tc.patience)
    gen = torch.Generator().manual_seed(tc.seed)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, tc.batch_size):
            xb = Xtr[perm[start : start + tc.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_mse = total / n
        val_mse = float(mse_per_sample(model, Xva).mean()) if len(Xva) else float("nan")
        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_legit_mse": val_mse,
                "lr": lr,
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        scheduler.step(val_mse)
        if stopper.step(val_mse, epoch):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if stopper.should_stop:
            break

    wall = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    with (run_dir / HISTORY_FILE).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        w.writeheader()
        w.writerows(history)
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": input_dim,
            "model_config": dataclasses.asdict(cfg.model),
            "best_epoch": stopper.best_epoch,
            "best_val_legit_mse": stopper.best,
            "seed": tc.seed,
        },
        run_dir / MODEL_FILE,
    )
    summary = {
        "epochs_run": len(history),
        "max_epochs": tc.epochs,
        "stopped_early": len(history) < tc.epochs,
        "best_epoch": stopper.best_epoch,
        "best_val_legit_mse": stopper.best,
        "final_train_mse": history[-1]["train_mse"],
        "final_lr": history[-1]["lr"],
        "n_train": int(n),
        "n_val_legit": int(len(Xva)),
        "input_dim": int(input_dim),
        "n_parameters": count_parameters(model),
        "seed": tc.seed,
        "wall_seconds": wall,
    }
    (run_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))

    from .plots import training_curves  # keep matplotlib out of the import path

    training_curves(run_dir / HISTORY_FILE, run_dir / "figures" / "training_curves.png")
    return model


def load_model(run_dir: Path) -> Autoencoder:
    """Rebuild the model from ``model.pt`` (no config file needed)."""
    ckpt = torch.load(Path(run_dir) / MODEL_FILE, map_location="cpu", weights_only=True)
    mc = ckpt["model_config"]
    mc["hidden_dims"] = tuple(mc["hidden_dims"])
    model = build_model(int(ckpt["input_dim"]), ModelConfig(**mc))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_history(run_dir: Path) -> list[dict[str, float]]:
    with (Path(run_dir) / HISTORY_FILE).open() as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / SUMMARY_FILE).read_text())
