import dataclasses

import pytest

from fraud_ae.config import ConfigError, load_config


def test_base_config_loads_and_validates(base_config_path):
    cfg = load_config(base_config_path)
    assert cfg.data.time_feature == "sincos"
    assert cfg.data.n_features == 31
    assert cfg.model.hidden_dims == (20, 10)
    assert cfg.model.latent_dim == 5
    assert cfg.threshold.target_fpr < cfg.eval.target_fpr  # margin below the 1 % criterion


def test_all_shipped_configs_load(project_root):
    for path in sorted((project_root / "configs").glob("*.toml")):
        load_config(path)


def test_no_time_ablation_has_29_inputs(project_root):
    cfg = load_config(project_root / "configs" / "ablation_no_time.toml")
    assert cfg.data.n_features == 29


def test_config_is_frozen(base_config_path):
    cfg = load_config(base_config_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.latent_dim = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_toml",
    [
        "[data]\nlegit_fractions = [0.5, 0.5, 0.5]\n",
        "[model]\nhidden_dims = [20, 10]\nlatent_dim = 10\n",
        "[score]\nmethod = 'cosine'\n",
        "[threshold]\ntarget_fpr = 1.5\n",
        "[data]\nno_such_key = 1\n",
        "[nonsense]\nx = 1\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path, bad_toml):
    p = tmp_path / "bad.toml"
    p.write_text(bad_toml)
    with pytest.raises(ConfigError):
        load_config(p)
