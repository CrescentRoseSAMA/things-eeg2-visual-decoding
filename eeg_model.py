"""Lightweight EEG encoders and the paired EEG–image training objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F


PAIR_TEMPERATURE = 0.07
MAX_LOGIT_SCALE = 100.0
LEARNABLE_TEMPERATURE_CONFIG = "ab_pool4_resproj_learn_temp"
COSINE_REGRESSION_CONFIG = "ab_pool4_resproj_cosine"
TRANSFER_CONFIG = "ab_pool4_resproj_cosine_transfer"
ELECTRODE_ATTENTION_CONFIG = "ab_pool4_resproj_cosine_electrode_attention"
EEGPROJECT_CONFIG = "eegproject_cosine"
CNN_COSINE_CONFIGS = (COSINE_REGRESSION_CONFIG, TRANSFER_CONFIG, ELECTRODE_ATTENTION_CONFIG)
COSINE_REGRESSION_CONFIGS = (*CNN_COSINE_CONFIGS, EEGPROJECT_CONFIG)
EEGPROJECT_DROPOUT = 0.3
COSINE_REGRESSION_WEIGHT = 0.25
SOFT_CLIP_WEIGHT = 0.30
SOFT_CLIP_TEMPERATURE = 0.10


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


class ResidualElectrodeAttention(nn.Module):
    """Mix electrode tokens while retaining each token's full time series."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(80)
        self.attention = nn.MultiheadAttention(80, 4, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, eeg):
        tokens = self.norm(eeg)
        mixed, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return eeg + self.alpha * mixed


class EEGProjectEncoder(nn.Module):
    """Project the flattened electrode-by-time grid with a residual MLP."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(17 * 80, 768)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(768, 768), nn.Dropout(EEGPROJECT_DROPOUT))
        self.norm = nn.LayerNorm(768)
        self.logit_scale = None

    def forward(self, eeg):
        projected = self.projection(eeg.flatten(1))
        return F.normalize(self.norm(projected + self.residual(projected)), dim=1)


class EEGEncoder(nn.Module):
    """Encode a batch of standardized 17-electrode, 80-sample epochs."""

    def __init__(self, config, *, adapter_count=1):
        super().__init__()
        overlap = config == "ab_pool8_stride4"
        learned = config == "ab_pool4_weighted"
        pointwise = config == "ab_pool4_pwmix"
        learn_temperature = config == LEARNABLE_TEMPERATURE_CONFIG
        transfer = config == TRANSFER_CONFIG
        attention = config == ELECTRODE_ATTENTION_CONFIG
        residual_projection = config == "ab_pool4_resproj" or config in CNN_COSINE_CONFIGS or learn_temperature
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
        self.logit_scale = None
        if learn_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / PAIR_TEMPERATURE)))

        self.electrode_adapters = None
        if transfer:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                self.electrode_adapters = nn.ModuleList(nn.Linear(17, 17, bias=False) for _ in range(adapter_count))
                for adapter in self.electrode_adapters:
                    adapter.weight.copy_(torch.eye(17))
        self.electrode_attention = None
        if attention:
            with torch.random.fork_rng(devices=[]):
                self.electrode_attention = ResidualElectrodeAttention()

    def forward(self, eeg, adapter_index=0):
        if self.electrode_adapters is not None:
            eeg = self.electrode_adapters[adapter_index](eeg.transpose(1, 2)).transpose(1, 2)
        if self.electrode_attention is not None:
            eeg = self.electrode_attention(eeg)
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


def contrastive_temperature(logit_scale):
    if logit_scale is None:
        return PAIR_TEMPERATURE
    return logit_scale.exp().clamp(max=MAX_LOGIT_SCALE).reciprocal()


def contrastive_loss(eeg_normalized, image, extra_images=None, *, logit_scale=None, cosine_weight=0.0,
                     soft_clip=False):
    """Symmetric InfoNCE with an optional cosine loss on the matched pairs."""
    temperature = contrastive_temperature(logit_scale)
    images_normalized = F.normalize(image, dim=1)
    similarities = eeg_normalized @ images_normalized.T
    logits = similarities / temperature
    targets = torch.arange(len(logits), device=logits.device)
    forward_logits = logits
    if extra_images is not None:
        extra = torch.bmm(extra_images, eeg_normalized.unsqueeze(2)).squeeze(2) / temperature
        forward_logits = torch.cat((logits, extra), dim=1)
    forward_loss = F.cross_entropy(forward_logits, targets)
    if soft_clip:
        with torch.no_grad():
            image_logits = images_normalized @ images_normalized.T / SOFT_CLIP_TEMPERATURE
            image_targets = image_logits.softmax(dim=1)
        forward_log_prob = forward_logits.log_softmax(dim=1)[:, :len(logits)]
        soft_loss = -(image_targets * forward_log_prob).sum(dim=1).mean()
        forward_loss = (1 - SOFT_CLIP_WEIGHT) * forward_loss + SOFT_CLIP_WEIGHT * soft_loss
    loss = (forward_loss + F.cross_entropy(logits.T, targets)) / 2
    if cosine_weight:
        return (1 - cosine_weight) * loss + cosine_weight * (1 - similarities.diagonal().mean())
    return loss


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

    def top5_loss(self, eeg, paired, ids, gallery):
        targets = self.targets[ids]
        positive = (eeg * F.normalize(self.sums[targets] - paired, dim=1)).sum(1)
        eligible = targets[:, None] != gallery
        # Use the last draw only when it needs to replace the masked target.
        eligible[:, -1] &= (~eligible[:, :-1]).any(1)
        candidates = eeg @ F.normalize(self.centers[gallery], dim=1).T
        fifth = candidates.masked_fill(~eligible, -torch.inf).topk(5, dim=1).values[:, -1]
        return ((fifth - positive) / PAIR_TEMPERATURE).relu().mean()
