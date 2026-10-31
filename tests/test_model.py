import torch

from fraud_ae.config import ModelConfig
from fraud_ae.model import build_model, count_parameters


def test_forward_shape_and_size():
    cfg = ModelConfig(hidden_dims=(20, 10), latent_dim=5)
    model = build_model(31, cfg)
    x = torch.randn(7, 31)
    assert model(x).shape == (7, 31)
    assert model.encode(x).shape == (7, 5)
    assert 1000 < count_parameters(model) < 5000


def test_seeded_models_are_identical():
    cfg = ModelConfig()
    torch.manual_seed(1)
    a = build_model(31, cfg)
    torch.manual_seed(1)
    b = build_model(31, cfg)
    x = torch.randn(3, 31)
    assert torch.equal(a(x), b(x))


def test_batch_norm_and_dropout_flags():
    model = build_model(29, ModelConfig(batch_norm=True, dropout=0.1))
    names = {type(m).__name__ for m in model.modules()}
    assert {"BatchNorm1d", "Dropout"} <= names


def test_vae_shapes_loss_and_seeded_reconstruction_nll():
    from fraud_ae.model import VariationalAutoencoder

    cfg = ModelConfig(kind="vae")
    torch.manual_seed(0)
    vae = build_model(31, cfg)
    assert isinstance(vae, VariationalAutoencoder)
    x = torch.randn(7, 31)
    assert vae(x).shape == (7, 31) and vae.encode(x).shape == (7, 5)
    mu, logvar = vae.encode_dist(x)
    assert mu.shape == logvar.shape == (7, 5)
    loss = vae.loss(x, torch.Generator().manual_seed(0))
    assert loss.ndim == 0 and torch.isfinite(loss)
    a = vae.reconstruction_nll(x, 4, torch.Generator().manual_seed(1))
    b = vae.reconstruction_nll(x, 4, torch.Generator().manual_seed(1))
    c = vae.reconstruction_nll(x, 4, torch.Generator().manual_seed(2))
    assert a.shape == (7,) and torch.equal(a, b) and not torch.equal(a, c)
    assert count_parameters(vae) > count_parameters(build_model(31, ModelConfig()))
    # the plain AE keeps the same loss interface
    ae = build_model(31, ModelConfig())
    assert torch.isclose(ae.loss(x), ((ae(x) - x) ** 2).mean())
