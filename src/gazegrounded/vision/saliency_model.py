"""Small image-to-saliency network: ResNet18 encoder + lightweight FPN-style decoder.

    screenshot (3 x H x W)
        -> ResNet18 layer2 / layer3 / layer4   (strides 8 / 16 / 32)
        -> 1x1 lateral convs to 64 channels, top-down upsample + add
        -> 3x3 conv -> 1 channel at stride 8
        -> bilinear upsample to H x W            -> saliency logits

~11.5 M parameters (almost all in the encoder), trainable on a laptop.

Loss
----
``loss = alpha * MSE(sigmoid(logits), target_max_normalised)
       + beta  * KL(target_distribution || softmax(logits))``

* KL compares *distributions* over the image (where does attention mass go?). It is the
  standard saliency objective and is insensitive to the absolute scale of the target.
* MSE on the sigmoid map anchors per-pixel magnitude and gives smoother gradients early in
  training when the softmax distribution is still nearly uniform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class SaliencyNet(nn.Module):
    def __init__(self, pretrained: bool = True, decoder_channels: int = 64):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2, self.layer3, self.layer4 = backbone.layer2, backbone.layer3, backbone.layer4
        c = decoder_channels
        self.lat2, self.lat3, self.lat4 = (
            nn.Conv2d(128, c, 1),
            nn.Conv2d(256, c, 1),
            nn.Conv2d(512, c, 1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )

    def encoder_parameters(self):
        for m in (self.stem, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f1 = self.stem(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        p = self.lat4(f4)
        p = _upsample_to(p, f3) + self.lat3(f3)
        p = _upsample_to(p, f2) + self.lat2(f2)
        logits = self.head(p)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def saliency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """``alpha * MSE + beta * KL`` (see module docstring). Shapes: (B, 1, H, W)."""
    b = logits.shape[0]
    mse = F.mse_loss(torch.sigmoid(logits), target)
    log_q = F.log_softmax(logits.view(b, -1), dim=1)
    p = target.view(b, -1)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    kl = (p * (torch.log(p + eps) - log_q)).sum(dim=1).mean()
    return alpha * mse + beta * kl


def logits_to_distribution(logits: torch.Tensor) -> torch.Tensor:
    b = logits.shape[0]
    return F.softmax(logits.view(b, -1), dim=1).view_as(logits)
