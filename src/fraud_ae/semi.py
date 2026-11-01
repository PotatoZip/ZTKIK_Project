"""Roadmap v2 Phase 9 – weakly-supervised autoencoders: the legit-only AE fine-tuned with a
handful of labelled frauds (ARCHITECTURE.md, "Supervised protocol").

``neg_ae`` (negative-example / margin loss): loss = MSE(legit batch) + λ · mean(relu(m − MSE(fraud
batch))) with the margin ``m`` = ``margin_quantile`` of the train-legit MSE under the warm-start
model; score stays the plain MSE, so the detector is still a reconstruction-error detector.

``sad_ae`` (Deep-SAD-style latent term, Ruff et al. 2020): loss = MSE(legit batch) +
η · (mean ‖z_l − c‖² + mean 1 / (‖z_f − c‖² + ε)) with the centre ``c`` = mean legit code under
the warm-start model and ε = ``SAD_EPS`` = 0.01 (caps the push-away term at 100 per row, so a
fraud whose code starts at the centre cannot blow up a step); score = ‖z − c‖².

Fine-tuning stops early and keeps the last finite parameters if a loss ever becomes non-finite
(``info["diverged"]``); the last partial legit batch of an epoch is dropped so that a handful of
legit rows is never paired with a full fraud batch.

Both start from the checkpoint in ``supervised.ae_run`` (never re-trained from scratch), run
``finetune_epochs`` passes over the legit rows with a fraud mini-batch drawn with replacement
next to every legit batch, and are seeded like everything else.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from .config import SupervisedConfig
from .model import Autoencoder
from .scoring import codes
from .train import mse_per_sample, seed_everything

SAD_EPS = 1e-2
LEGIT_BATCH = 512


def finetune(
    ae: Autoencoder,
    cfg: SupervisedConfig,
    Xl: np.ndarray,
    Xf: np.ndarray,
    seed: int,
    mode: str = "margin",
) -> tuple[Autoencoder, dict[str, Any]]:
    if mode not in ("margin", "sad"):
        raise ValueError("mode must be 'margin' or 'sad'")
    if len(Xf) == 0:
        raise ValueError("fine-tuning needs at least one labelled fraud")
    seed_everything(seed)
    model = copy.deepcopy(ae)
    Xl_t = torch.as_tensor(np.asarray(Xl), dtype=torch.float32)
    Xf_t = torch.as_tensor(np.asarray(Xf), dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        margin = float(torch.quantile(mse_per_sample(model, Xl_t), cfg.margin_quantile))
        centre = torch.as_tensor(codes(model, Xl).mean(axis=0), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    gen = torch.Generator().manual_seed(seed)
    n_l, bf = len(Xl_t), cfg.fraud_per_batch
    bl = min(LEGIT_BATCH, n_l)
    diverged = False
    model.train()
    for _ in range(cfg.finetune_epochs):
        perm = torch.randperm(n_l, generator=gen)
        for start in range(0, n_l - bl + 1, bl):  # drop the last partial legit batch
            xl = Xl_t[perm[start : start + bl]]
            xf = Xf_t[torch.randint(len(Xf_t), (bf,), generator=gen)]
            opt.zero_grad(set_to_none=True)
            loss = ((model(xl) - xl) ** 2).mean()
            if mode == "margin":
                mse_f = ((model(xf) - xf) ** 2).mean(dim=1)
                loss = loss + cfg.margin_weight * torch.relu(margin - mse_f).mean()
            else:
                d_l = ((model.encode(xl) - centre) ** 2).sum(dim=1)
                d_f = ((model.encode(xf) - centre) ** 2).sum(dim=1)
                loss = loss + cfg.sad_weight * (d_l.mean() + (1.0 / (d_f + SAD_EPS)).mean())
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            opt.step()
        if diverged:
            break
    model.eval()
    info = {
        "mode": mode,
        "margin": margin,
        "centre": centre.numpy(),
        "n_frauds": int(len(Xf)),
        "diverged": diverged,
    }
    return model, info


@torch.no_grad()
def semi_scores(model: Autoencoder, X: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """``neg_ae``: MSE; ``sad_ae``: squared distance of the code to the legit centre."""
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if info["mode"] == "margin":
        return mse_per_sample(model, xt).double().numpy()
    c = torch.as_tensor(info["centre"], dtype=torch.float32)
    return (
        torch.cat([((model.encode(xb) - c) ** 2).sum(dim=1) for xb in xt.split(8192)])
        .double()
        .numpy()
    )
