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
