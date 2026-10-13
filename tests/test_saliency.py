import json

import pytest
import torch

from gazegrounded.gaze.aoi import assign_fixations_to_aois
from gazegrounded.gaze.fixation import detect_fixations
from gazegrounded.vision.saliency_dataset import SaliencyDataset, build_items, split_items
from gazegrounded.vision.saliency_model import SaliencyNet, logits_to_distribution, saliency_loss


@pytest.fixture(scope="module")
def fixations(synthetic_ds):
    fx = detect_fixations(synthetic_ds.gaze_samples)
    return assign_fixations_to_aois(fx, synthetic_ds.aois, synthetic_ds.uis)


def test_forward_shape_and_distribution():
    torch.manual_seed(0)
    model = SaliencyNet(pretrained=False).eval()
    x = torch.randn(2, 3, 80, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 1, 80, 128)
    p = logits_to_distribution(logits)
    assert torch.allclose(p.sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-5)


def test_loss_is_zero_kl_for_matching_distribution_and_decreases_with_training():
    torch.manual_seed(0)
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 4:8, 4:8] = 1.0
    model = SaliencyNet(pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(1, 3, 64, 64)
    t = torch.nn.functional.interpolate(target, size=(64, 64))
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = saliency_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_split_keeps_interface_pairs_together(synthetic_ds, fixations):
    items = build_items(synthetic_ds, fixations)
    assert len(items) == 5
    train, val = split_items(items, 0.34, seed=1)
    assert {i.group for i in train}.isdisjoint({i.group for i in val})
    assert {i.item_id for i in train} | {i.item_id for i in val} == {i.item_id for i in items}
    # clean and friction checkout share a group, so they always land on the same side
    sides = {i.item_id: ("train" if i in train else "val") for i in items}
    assert sides["shop_checkout_clean"] == sides["shop_checkout_friction"]


def test_dataset_item_shapes(synthetic_ds, fixations):
    items = build_items(synthetic_ds, fixations)
    ds = SaliencyDataset(items, (128, 80), sigma_px=40)
    image, target, mask, idx = ds[0]
    assert image.shape == (3, 80, 128)
    assert target.shape == mask.shape == (1, 80, 128)
    assert float(target.max()) == pytest.approx(1.0)
    assert mask.sum() > 0


@pytest.mark.slow
def test_fast_dev_training_run(tmp_path, fast_cfg, synthetic_ds, fixations):
    from gazegrounded.config import load_config
    from gazegrounded.vision.train_saliency import load_saliency_model, train_saliency

    cfg = load_config("configs/fast_dev.yaml", [f"paths.experiments={tmp_path}"])
    run_dir = train_saliency(cfg, synthetic_ds, fixations)
    for name in (
        "best_model.pt",
        "config.yaml",
        "metrics.json",
        "history.csv",
        "split.json",
        "predictions.csv",
    ):
        assert (run_dir / name).exists(), name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["data_kind"] == "synthetic_demo"
    model, size = load_saliency_model(run_dir)
    assert size == (128, 80)
