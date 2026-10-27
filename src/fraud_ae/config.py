"""Experiment configuration: frozen dataclasses loaded from TOML (stdlib ``tomllib``).

A config file is the single source of truth for a run. Each run directory receives a
verbatim snapshot of the file it was started from (see ``train.py``).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

TimeFeature = Literal["sincos", "drop"]
Activation = Literal["elu", "relu", "tanh"]
ScoreMethod = Literal["mse", "standardized", "mahalanobis"]
ThresholdStrategy = Literal["quantile", "youden", "max_f1"]
SplitStrategy = Literal["random", "temporal"]


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    zip_path: str = "dataset.zip"
    raw_csv: str = "data/raw/creditcard.csv"
    processed_dir: str = "data/processed"
    expected_sha256: str = ""
    drop_duplicates: bool = True
    log_amount: bool = True
    time_feature: TimeFeature = "sincos"
    legit_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    fraud_val_fraction: float = 0.5
    seed: int = 42
    split: SplitStrategy = "random"  # "temporal": train on the earliest transactions

    @property
    def n_features(self) -> int:
        """Model input width: 28 V-features + Amount (+ sin/cos hour)."""
        return 28 + 1 + (2 if self.time_feature == "sincos" else 0)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (20, 10)
    latent_dim: int = 5
    activation: Activation = "elu"
    batch_norm: bool = False
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    seed: int = 42
    num_threads: int = 8


@dataclass(frozen=True)
class ScoreConfig:
    method: ScoreMethod = "mse"


@dataclass(frozen=True)
class ThresholdConfig:
    strategy: ThresholdStrategy = "quantile"
    target_fpr: float = 0.009


@dataclass(frozen=True)
class EvalConfig:
    fpr_grid: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02)
    target_recall: float = 0.85
    target_fpr: float = 0.01
    bootstrap_samples: int = 1000


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path")
        return d


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "score": ScoreConfig,
    "threshold": ThresholdConfig,
    "eval": EvalConfig,
}


def _build(section_cls: type, raw: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(section_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys: {sorted(unknown)}")
    # TOML arrays arrive as lists; dataclasses are frozen so store tuples.
    coerced = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return section_cls(**coerced)


def validate(cfg: Config) -> None:
    """Cross-field checks that TOML typing alone cannot express."""
    d, m, t, th, ev = cfg.data, cfg.model, cfg.train, cfg.threshold, cfg.eval
    if len(d.legit_fractions) != 3 or abs(sum(d.legit_fractions) - 1.0) > 1e-9:
        raise ConfigError("data.legit_fractions must be three numbers summing to 1")
    if any(f <= 0 for f in d.legit_fractions):
        raise ConfigError("data.legit_fractions must all be positive")
    if not 0.0 <= d.fraud_val_fraction <= 1.0:
        raise ConfigError("data.fraud_val_fraction must be in [0, 1]")
    if d.time_feature not in ("sincos", "drop"):
        raise ConfigError(f"data.time_feature invalid: {d.time_feature!r}")
    if d.split not in ("random", "temporal"):
        raise ConfigError(f"data.split invalid: {d.split!r}")
    if not m.hidden_dims or any(h <= 0 for h in m.hidden_dims):
        raise ConfigError("model.hidden_dims must be a non-empty list of positive ints")
    if not 0 < m.latent_dim < m.hidden_dims[-1]:
        raise ConfigError("model.latent_dim must be positive and smaller than the last hidden dim")
    if m.hidden_dims[0] > d.n_features:
        raise ConfigError("model.hidden_dims[0] must not exceed the number of input features")
    if m.activation not in ("elu", "relu", "tanh"):
        raise ConfigError(f"model.activation invalid: {m.activation!r}")
    if not 0.0 <= m.dropout < 1.0:
        raise ConfigError("model.dropout must be in [0, 1)")
    if t.epochs <= 0 or t.batch_size <= 0 or t.patience <= 0 or t.lr <= 0:
        raise ConfigError("train.epochs, batch_size, patience and lr must be positive")
    if cfg.score.method not in ("mse", "standardized", "mahalanobis"):
        raise ConfigError(f"score.method invalid: {cfg.score.method!r}")
    if th.strategy not in ("quantile", "youden", "max_f1"):
        raise ConfigError(f"threshold.strategy invalid: {th.strategy!r}")
    if not 0.0 < th.target_fpr < 1.0:
        raise ConfigError("threshold.target_fpr must be in (0, 1)")
    if not 0.0 < ev.target_fpr < 1.0 or not 0.0 < ev.target_recall <= 1.0:
        raise ConfigError("eval.target_fpr / target_recall out of range")
    if any(not 0.0 < f < 1.0 for f in ev.fpr_grid):
        raise ConfigError("eval.fpr_grid values must be in (0, 1)")


def load_config(path: str | Path) -> Config:
    """Parse a TOML file into a validated :class:`Config`."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown top-level sections: {sorted(unknown)}")
    sections = {name: _build(cls, raw.get(name, {}), name) for name, cls in _SECTIONS.items()}
    cfg = Config(**sections, source_path=str(path))
    validate(cfg)
    return cfg
