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
        "[score]\nmethod = 'recon_prob'\n",  # needs model.kind = 'vae'
        "[data]\ntime_feature = 'drop'\n[threshold]\nstrategy = 'quantile_by_hour'\n",
        "[score]\nn_samples = 0\n",
        "[threshold]\nnight_hours = [6, 3]\n",
        "[model]\nkind = 'gan'\n",
    ],
)
def test_invalid_configs_are_rejected(tmp_path, bad_toml):
    p = tmp_path / "bad.toml"
    p.write_text(bad_toml)
    with pytest.raises(ConfigError):
        load_config(p)


def test_phase8_configs_carry_their_method(project_root):
    vae = load_config(project_root / "configs" / "v2_unsup_vae.toml")
    assert (
        vae.model.kind == "vae" and vae.score.method == "recon_prob" and vae.score.n_samples == 10
    )
    hour = load_config(project_root / "configs" / "v2_unsup_by_hour.toml")
    assert hour.threshold.strategy == "quantile_by_hour" and hour.threshold.night_hours == (0, 6)
    gmm = load_config(project_root / "configs" / "v2_unsup_latent_gmm8.toml")
    assert gmm.score.method == "latent_gmm" and gmm.score.gmm_components == 8
    t80 = load_config(project_root / "configs" / "v2_unsup_train80.toml")
    assert t80.data.legit_fractions == (0.8, 0.1, 0.1) and t80.data.fraud_val_fraction == 0.0
