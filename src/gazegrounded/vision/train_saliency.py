"""Training loop for :class:`~gazegrounded.vision.saliency_model.SaliencyNet`.

Writes ``experiments/<timestamp>-saliency/`` with ``config.yaml``, ``split.json``,
``history.csv``, ``metrics.json``, ``predictions.csv`` (per validation item) and
``best_model.pt`` (state dict of the epoch with the lowest validation loss).

On the synthetic demo (five screenshots) this run is a *software test* of the training code;
its numbers are meaningless as a model evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gazegrounded.config import config_to_dict
from gazegrounded.utils import (
    get_logger,
    make_run_dir,
    resolve_device,
    save_json,
    save_json_safe,
    save_yaml,
    set_seed,
)
from gazegrounded.vision.evaluate_saliency import evaluate, summarize
from gazegrounded.vision.saliency_dataset import SaliencyDataset, build_items, split_items
from gazegrounded.vision.saliency_model import SaliencyNet, saliency_loss

log = get_logger(__name__)


def _loader(ds: SaliencyDataset, batch_size: int, shuffle: bool, workers: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=g)


def train_saliency(cfg, ds, fixations: pd.DataFrame, run_name: str = "saliency") -> Path:
    s = cfg.saliency
    set_seed(cfg.seed)
    device = resolve_device(s.device)
    items = build_items(ds, fixations)
    train_items, val_items = split_items(items, s.val_fraction, cfg.seed)
    if not train_items or not val_items:
        raise ValueError(f"Need >=2 interface groups to train/validate, got {len(items)} items")
    size = (s.image_width, s.image_height)
    train_ds = SaliencyDataset(train_items, size, cfg.gaze.heatmap_sigma_px)
    val_ds = SaliencyDataset(val_items, size, cfg.gaze.heatmap_sigma_px)
    train_dl = _loader(train_ds, s.batch_size, True, s.num_workers, cfg.seed)
    val_dl = _loader(val_ds, s.batch_size, False, s.num_workers, cfg.seed)

    model = SaliencyNet(pretrained=s.pretrained).to(device)
    if s.freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=s.learning_rate,
        weight_decay=s.weight_decay,
    )

    run_dir = make_run_dir(cfg.experiments_dir(), run_name)
    save_yaml(config_to_dict(cfg), run_dir / "config.yaml")
    save_json(
        {
            "train": [it.item_id for it in train_items],
            "val": [it.item_id for it in val_items],
            "group_key": "pair_id (else ui_id)",
            "data_kind": cfg.data.data_kind,
        },
        run_dir / "split.json",
    )
    log.info(
        "Saliency training on %s: %d train / %d val screenshots, %d epochs -> %s",
        device,
        len(train_ds),
        len(val_ds),
        s.epochs,
        run_dir,
    )

    max_batches = 2 if s.fast_dev_run else None
    best, best_epoch, patience, history = float("inf"), -1, 0, []
    for epoch in range(1, s.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for b, (images, targets, _, _) in enumerate(train_dl):
            if max_batches is not None and b >= max_batches:
                break
            opt.zero_grad()
            loss = saliency_loss(
                model(images.to(device)), targets.to(device), s.loss_alpha_mse, s.loss_beta_kl
            )
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_losses = [
                saliency_loss(
                    model(images.to(device)), targets.to(device), s.loss_alpha_mse, s.loss_beta_kl
                ).item()
                for images, targets, _, _ in val_dl
            ]
        val_metrics = summarize(evaluate(model, val_dl, device))
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": sum(val_losses) / len(val_losses),
            "seconds": time.time() - t0,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "n_items"},
        }
        history.append(row)
        log.info(
            "epoch %d  train %.4f  val %.4f  val CC %.3f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row.get("val_cc_mean", float("nan")),
        )
        if row["val_loss"] < best - 1e-6:
            best, best_epoch, patience = row["val_loss"], epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "image_size": size,
                    "pretrained": s.pretrained,
                },
                run_dir / "best_model.pt",
            )
        else:
            patience += 1
            if patience >= s.patience:
                log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device)["state_dict"])
    per_item = evaluate(model, val_dl, device)
    per_item.to_csv(run_dir / "predictions.csv", index=False)
    save_json_safe(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "device": device,
            "data_kind": cfg.data.data_kind,
            **summarize(per_item),
            "note": "synthetic demo: software test only" if cfg.is_synthetic else "",
        },
        run_dir / "metrics.json",
    )
    _training_figures(cfg, run_dir, history, model, val_ds, device)
    return run_dir


def load_saliency_model(run_dir: Path, device: str = "cpu") -> tuple[SaliencyNet, tuple[int, int]]:
    ckpt = torch.load(Path(run_dir) / "best_model.pt", map_location=device)
    model = SaliencyNet(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), tuple(ckpt["image_size"])


def _training_figures(cfg, run_dir, history, model, val_ds, device) -> None:
    from gazegrounded.vision.image_utils import tensor_to_image
    from gazegrounded.vision.saliency_model import logits_to_distribution
    from gazegrounded.visualization import style
    from gazegrounded.visualization.style import plt, save_figure

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(h["epoch"], h["train_loss"], color=style.CATEGORICAL[0], lw=2, label="train")
    ax.plot(h["epoch"], h["val_loss"], color=style.CATEGORICAL[1], lw=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE + KL)")
    ax.grid(True)
    ax.legend()
    ax.set_title("Saliency model training", loc="left")
    fig.tight_layout()
    save_figure(fig, run_dir / "loss_curve.png", cfg.is_synthetic)

    n = min(3, len(val_ds))
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.4 * n), squeeze=False)
    model.eval()
    for i in range(n):
        image, target, _, _ = val_ds[i]
        with torch.no_grad():
            pred = logits_to_distribution(model(image[None].to(device)))[0, 0].cpu().numpy()
        for ax, arr, title in zip(
            axes[i],
            [tensor_to_image(image), target[0].numpy(), pred],
            ["screenshot (held out)", "human gaze map", "predicted saliency"],
            strict=True,
        ):
            ax.imshow(arr, cmap=style.BLUES if arr.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{val_ds.items[i].item_id}: {title}" if title.startswith("screen") else title,
                fontsize=9,
            )
    fig.tight_layout()
    save_figure(fig, run_dir / "val_predictions.png", cfg.is_synthetic)
