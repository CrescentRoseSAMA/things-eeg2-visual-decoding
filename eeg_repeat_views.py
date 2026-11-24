"""Three-of-four EEG averages and an independent, epoch-addressable sampling stream."""
import numpy as np
import torch


def omitted_means(raw):
    return np.stack([raw[:, [i for i in range(4) if i != omitted]].mean(axis=1)
                     for omitted in range(4)], axis=1).astype(np.float32)


def sample_epoch(full, views, seed, epoch):
    rng = np.random.default_rng([seed, epoch, 304])
    draws = torch.as_tensor(rng.integers(8, size=len(full)), device=full.device)
    alternative = views[torch.arange(len(full), device=full.device), draws % 4]
    return torch.where((draws < 4)[:, None, None], alternative, full)
