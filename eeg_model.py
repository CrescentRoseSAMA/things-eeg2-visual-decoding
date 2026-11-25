"""Lightweight EEG encoders and the paired EEG–image training objective."""

import torch
from torch import nn
from torch.nn import functional as F


class WeightedTimePool(nn.Module):
    """Mix four successive samples with one learned distribution per channel."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(32, 4))

    def forward(self, x):
        weights = self.logits.softmax(dim=1)[None, :, None, None, :]
        return (x.unfold(-1, 4, 4) * weights).sum(dim=-1)


class ResidualPointwiseMixing(nn.Module):
    """Mix temporal feature channels before averaging adjacent samples."""

    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ELU())
        self.alpha = nn.Parameter(torch.zeros(1, 32, 1, 1))

    def forward(self, x):
        return x + self.alpha * self.branch(x)


class EEGEncoder(nn.Module):
    """Encode a batch of standardized 17-electrode, 80-sample epochs."""

    def __init__(self, config):
        super().__init__()
        overlap = config == "ab_pool8_stride4"
        learned = config == "ab_pool4_weighted"
        pointwise = config == "ab_pool4_pwmix"
        residual_projection = config == "ab_pool4_resproj"
        finer_time = config in {"b_pool4", "ab_pool4"} or overlap or learned or pointwise or residual_projection
        if config == "b_pool4":
            config = "b"
        elif finer_time:
            config = "ab"
        if config not in {"baseline", "a", "b", "ab"}:
            raise ValueError(f"Unknown EEG encoder configuration: {config}")

        # Each temporal filter is shared across all electrodes.
        scales = [(5, 8), (11, 16), (21, 8)] if config in {"a", "ab"} else [(11, 32)]
        self.temporal = nn.ModuleList(
            nn.Conv2d(1, width, (1, kernel), padding=(0, kernel // 2), bias=False)
            for kernel, width in scales
        )
        self.electrode_gate = None
        if config in {"b", "ab"}:
            self.electrode_gate = nn.Sequential(
                nn.Linear(17, 8), nn.ReLU(), nn.Linear(8, 17), nn.Sigmoid()
            )

        # One spatial filter per temporal feature mixes the 17 electrodes.
        self.readout = nn.Sequential(
            nn.Conv2d(32, 32, (17, 1), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(32 * 10, 768),
        )
        if finer_time:
            self.readout[3] = (nn.AvgPool2d((1, 8), stride=(1, 4), padding=(0, 2),
                                            count_include_pad=False) if overlap else nn.AvgPool2d((1, 4)))
            if learned:
                self.readout[3] = WeightedTimePool()
            original = self.readout[-1]
            # Each 8-point mean equals the mean of its two 4-point means.
            # The overlapping control uses the identical wider head initialization.
            with torch.random.fork_rng(devices=[]):
                finer = nn.Linear(32 * 20, 768)
            with torch.no_grad():
                finer.weight.copy_(original.weight.repeat_interleave(2, dim=1) / 2)
                finer.bias.copy_(original.bias)
            self.readout[-1] = finer
        if pointwise:
            with torch.random.fork_rng(devices=[]):
                self.readout.insert(3, ResidualPointwiseMixing())
        self.residual_projection = None
        if residual_projection:
            with torch.random.fork_rng(devices=[]):
                self.residual_projection = nn.Sequential(
                    nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                    nn.Dropout(0.5), nn.Linear(256, 768),
                )
            nn.init.zeros_(self.residual_projection[-1].weight)
            nn.init.zeros_(self.residual_projection[-1].bias)

    def forward(self, eeg):
        signal = eeg.unsqueeze(1)
        if len(self.temporal) == 1:
            features = self.temporal[0](signal)
        else:
            features = torch.cat([conv(signal) for conv in self.temporal], dim=1)
        if self.electrode_gate is not None:
            summary = features.mean(dim=(1, 3))
            gains = self.electrode_gate(summary)
            features = features * gains[:, None, :, None]
        z = self.readout(features)
        if self.residual_projection is not None:
            z = z + self.residual_projection(z)
        return F.normalize(z, dim=1)


def contrastive_loss(eeg_normalized, image, extra_images=None):
    """Symmetric InfoNCE using the matching diagonal of the paired minibatch."""
    images_normalized = F.normalize(image, dim=1)
    logits = (eeg_normalized @ images_normalized.T) / 0.07
    targets = torch.arange(len(logits), device=logits.device)
    forward_logits = logits
    if extra_images is not None:
        extra = torch.bmm(extra_images, eeg_normalized.unsqueeze(2)).squeeze(2) / 0.07
        forward_logits = torch.cat((logits, extra), dim=1)
    return (F.cross_entropy(forward_logits, targets) + F.cross_entropy(logits.T, targets)) / 2


class ClassPrototypes:
    """Class means of fixed training images, excluding the paired image at scoring."""

    def __init__(self, images, labels):
        labels = torch.as_tensor(labels, device=images.device)
        if labels.shape != (len(images),):
            raise ValueError("Expected one class per training image")
        order = labels.argsort(stable=True)
        _, self.targets, counts = labels.unique(sorted=True, return_inverse=True, return_counts=True)
        if not (counts == 10).all():
            raise ValueError("Expected ten training images per class")
        self.sums = images[order].reshape(-1, 10, images.shape[1]).sum(1)
        self.centers = self.sums / 10

    def loss(self, eeg, paired, ids):
        targets = self.targets[ids]
        own = ((self.sums[targets] - paired) * eeg).sum(1, keepdim=True) / 9
        logits = (eeg @ self.centers.T).scatter(1, targets[:, None], own)
        return F.cross_entropy(logits / 0.07, targets)
