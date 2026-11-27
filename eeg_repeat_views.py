"""Trial subset means with independent, epoch-addressable sampling streams."""
import numpy as np
import torch


TRIAL_SAMPLING = "uniform k in {2,3,4}; sample trials without replacement"
SINGLE_TRIAL_SAMPLING = "uniform k in {1,2,3,4}; sample trials without replacement"
TRIAL_NOISE_STD = 0.10
TRIAL_CHANNEL_DROPOUT = 0.05


def omitted_means(raw):
    return np.stack([raw[:, [i for i in range(4) if i != omitted]].mean(axis=1)
                     for omitted in range(4)], axis=1).astype(np.float32)


def sample_epoch(full, views, seed, epoch):
    rng = np.random.default_rng([seed, epoch, 304])
    draws = torch.as_tensor(rng.integers(8, size=len(full)), device=full.device)
    alternative = views[torch.arange(len(full), device=full.device), draws % 4]
    return torch.where((draws < 4)[:, None, None], alternative, full)


def sample_subset_epoch(full, repeats, seed, epoch, *, min_trials=2):
    rng = np.random.default_rng([seed, epoch, 604])
    sizes = rng.integers(min_trials, 5, size=len(full))
    permutation = rng.random((len(full), 4)).argsort(axis=1)
    weights = torch.as_tensor((permutation < sizes[:, None]) / sizes[:, None],
                              dtype=full.dtype, device=full.device)
    averaged = torch.bmm(weights.unsqueeze(1), repeats.flatten(2)).reshape_as(full)
    use_full = torch.as_tensor(sizes == 4, device=full.device)
    return torch.where(use_full[:, None, None], full, averaged)


def perturb_trial_epoch(eeg, seed, epoch, *, noise_std=TRIAL_NOISE_STD):
    noise_seed = int(np.random.SeedSequence([seed, epoch, 611]).generate_state(1, dtype=np.uint64)[0])
    generator = torch.Generator(device=eeg.device).manual_seed(noise_seed)
    noise = torch.randn(eeg.shape, dtype=eeg.dtype, device=eeg.device, generator=generator)
    drop = torch.rand(eeg.shape[:2], device=eeg.device, generator=generator) < TRIAL_CHANNEL_DROPOUT
    return noise.mul_(noise_std).add_(eeg).masked_fill_(drop.unsqueeze(2), 0)
