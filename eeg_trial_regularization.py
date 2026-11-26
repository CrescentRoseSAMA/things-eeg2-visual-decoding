"""Paired development pilot for single trials, input perturbations, and weight EMA."""
import numpy as np
import torch

from eeg_csls_replication import ADJUSTED, CANDIDATE, scoring_arms, training_stats
from eeg_data import load_subject
from eeg_gemini_iterations import (BASE, BATCH, EPOCHS, PILOT, contrast, design_for, freeze, pairs_as_lists,
                                   read_development, result_row)
from eeg_pooling_dev import evaluate_checkpoint
from eeg_repeat_views import SINGLE_TRIAL_SAMPLING, TRIAL_CHANNEL_DROPOUT, TRIAL_NOISE_STD, TRIAL_SAMPLING
from eeg_train import DEVICE, EMA_DECAY, EMA_FIRST_EPOCH, OUT, run_dir, save_json, tensor, train, write_log
from eeg_trial_augmentation import (CANDIDATE_ARM, OUTPUT, checked_trials, gate, protocol_for, stored_csls,
                                    top5_wins)


DIRECTORY = BASE / "round-11" / "pilot"
K1 = "single_trial_enabled"
K1_EMA = "single_trial_enabled_ema"
PERTURBED = "single_trial_enabled_perturbed"
FULL = "single_trial_enabled_perturbed_ema"
COMPARISONS = ((K1, CANDIDATE_ARM), (PERTURBED, CANDIDATE_ARM), (K1_EMA, CANDIDATE_ARM), (FULL, CANDIDATE_ARM),
               (PERTURBED, K1), (K1_EMA, K1), (FULL, PERTURBED))


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    control_stage = (OUTPUT / "pilot").relative_to(OUT).as_posix()
    protocol = protocol_for(PILOT)
    protocol.update(candidate=FULL, reference=CANDIDATE_ARM, reference_stage=control_stage, split_seed=2026,
                    comparisons=pairs_as_lists(COMPARISONS), training_eeg=SINGLE_TRIAL_SAMPLING,
                    trial_noise_std=TRIAL_NOISE_STD, trial_channel_dropout=TRIAL_CHANNEL_DROPOUT,
                    perturbation="Gaussian noise in standardized units, then zero entire dropped channels",
                    perturbation_rng="independent torch generator seeded by NumPy SeedSequence([seed, epoch, 611])",
                    ema_decay=EMA_DECAY, ema_first_epoch=EMA_FIRST_EPOCH,
                    ema="update once per epoch, initialize at epoch 51, copy current batch-norm buffers",
                    gate="full arm vs round 06: Top-5 >= +1pp and Top-1 >= 0pp; at least 3/4 improve Top-5",
                    query="standard four-repeat average, fixed 200 templates and 1000 queries", new_fits=8)
    freeze(DIRECTORY / "protocol.json", protocol)
    design = design_for(2026)
    indices, templates, labels = (design[key] for key in ("dev_train", "dev_centers", "dev_labels"))
    centers = tensor(templates)
    rows = []
    for subject, seed in PILOT:
        reference = run_dir(control_stage, CANDIDATE, subject, seed)
        read_development(reference / "validation.npz", design)
        base_stats = training_stats(reference, CANDIDATE, subject, seed)
        if base_stats.get("training_eeg") != TRIAL_SAMPLING:
            raise ValueError(f"Expected the round 06 trial augmentation control: {reference}")
        control = torch.load(reference / "checkpoint.pt", map_location=DEVICE)
        ranks = {CANDIDATE_ARM: stored_csls(reference / "csls.npz", labels)}
        stats = {CANDIDATE_ARM: base_stats}
        full = load_subject(subject)
        repeats = checked_trials(subject, indices, full)
        bank = full[indices]
        query = tensor(full[design["dev_query"]])
        for arm, averaged_arm, perturb in ((K1, K1_EMA, False), (PERTURBED, FULL, True)):
            stage = (DIRECTORY / arm).relative_to(OUT).as_posix()
            train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, design, split="dev", trial_repeats=repeats,
                  single_trials=True, perturb_trials=perturb, weight_ema=True)
            folder = run_dir(stage, CANDIDATE, subject, seed)
            profile = training_stats(folder, CANDIDATE, subject, seed)
            if (profile.get("training_eeg") != SINGLE_TRIAL_SAMPLING or profile["parameters"] != 889353
                    or profile.get("ema_decay") != EMA_DECAY or profile.get("ema_first_epoch") != EMA_FIRST_EPOCH
                    or profile.get("trial_noise_std", 0) != (TRIAL_NOISE_STD if perturb else 0)
                    or profile.get("trial_channel_dropout", 0) != (TRIAL_CHANNEL_DROPOUT if perturb else 0)):
                raise ValueError(f"Unexpected trial regularization settings: {folder}")
            ema = torch.load(folder / "ema.pt", map_location="cpu")
            if ema["ema_updates"] != EPOCHS - EMA_FIRST_EPOCH + 1:
                raise ValueError(f"Incomplete weight averaging: {folder}")
            del ema
            for name, checkpoint, prefix in ((arm, "checkpoint.pt", ""), (averaged_arm, "ema.pt", "ema_")):
                validation = folder / f"{prefix}validation.npz"
                _, mean, std = evaluate_checkpoint(CANDIDATE, subject, seed, design, query, centers, validation,
                                                   stage, checkpoint_name=checkpoint)
                torch.testing.assert_close((mean, std), (control["mean"], control["std"]), rtol=0, atol=0)
                read_development(validation, design)
                with np.load(validation) as raw:
                    ranks[name] = scoring_arms(folder, raw["scores"], labels, templates, bank,
                                               destination=folder / f"{prefix}csls.npz",
                                               checkpoint_name=checkpoint)[ADJUSTED]
                stats[name] = profile
        for candidate, reference_arm in COMPARISONS:
            rows.append(result_row(subject, seed, candidate, reference_arm, ranks[candidate], ranks[reference_arm],
                                   stats))
    summaries = {}
    for candidate, reference in COMPARISONS:
        subset = [row for row in rows if (row["candidate"], row["control"]) == (candidate, reference)]
        summary = contrast(subset)
        del summary["pilot_pass"], summary["formal_gate_pass"]
        summary["top5_subject_wins"] = top5_wins(subset)
        if candidate == FULL and reference == CANDIDATE_ARM:
            summary["gate_pass"] = gate("pilot", subset)
        summaries[f"{candidate}_vs_{reference}"] = summary
    write_log(DIRECTORY / "paired_runs.csv", rows)
    result = dict(comparisons=summaries, new_fits=8, official_test_used=False)
    save_json(DIRECTORY / "summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
