"""Compare two training noise levels with fixed QB-Norm scoring."""

import numpy as np
import torch

from eeg_csls_replication import CANDIDATE, training_stats
from eeg_data import load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import (BASE, BATCH, EPOCHS, PILOT, contrast, design_for, freeze, pairs_as_lists,
                                   read_development, result_row)
from eeg_pooling_dev import evaluate_checkpoint
from eeg_qb_csls_pilot import CANDIDATE_ARM as ROUND06_QB, DIRECTORY as ROUND13
from eeg_querybank import BETA, qb_scores, querybank_statistics
from eeg_repeat_views import SINGLE_TRIAL_SAMPLING, TRIAL_CHANNEL_DROPOUT, TRIAL_NOISE_STD
from eeg_train import DEVICE, EMA_DECAY, EMA_FIRST_EPOCH, LR, OUT, load_json, model_for, run_dir, save_json, score, tensor, train, write_log
from eeg_trial_augmentation import OUTPUT, checked_trials, gate, top5_wins
from eeg_trial_regularization import DIRECTORY as ROUND11, PERTURBED


DIRECTORY = BASE / "round-14" / "pilot"
NOISE_STD = 0.05
CANDIDATE_ARM = "noise005_ema_qb"
PARENT_ARM = "noise010_ema_qb"


def score_ema(folder, destination, design, train_eeg, query, centers, norms, reference):
    read_development(folder / "ema_validation.npz", design)
    checkpoint = torch.load(folder / "ema.pt", map_location=DEVICE)
    if checkpoint["ema_updates"] != EPOCHS - EMA_FIRST_EPOCH + 1:
        raise ValueError(f"Incomplete EMA updates: {folder}")
    model = model_for(CANDIDATE)
    model.load_state_dict(checkpoint["model"], strict=True)
    mean, std = checkpoint["mean"], checkpoint["std"]
    torch.testing.assert_close((mean, std), (reference["mean"], reference["std"]), rtol=0, atol=0)
    bank = score(model, (train_eeg - mean) / std, centers, BATCH).cpu().numpy() / norms
    raw = score(model, (query - mean) / std, centers, BATCH).cpu().numpy()
    with np.load(folder / "ema_validation.npz") as validation:
        np.testing.assert_allclose(raw, validation["scores"], atol=2e-6, rtol=1e-5)
    cosine = raw / norms
    stats = querybank_statistics(bank)
    adjusted, _ = qb_scores(cosine, stats)
    labels = design["dev_labels"]
    ranks = ranks_and_accuracy(adjusted, labels)[0]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "qb_scores.npz", query_rows=design["dev_query"], labels=labels,
                        scores=adjusted, ranks=ranks, bank_cosine=bank, normalized_scores=cosine, **stats)
    return ranks


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    stage = DIRECTORY.relative_to(OUT).as_posix()
    parent_stage = (ROUND11 / PERTURBED).relative_to(OUT).as_posix()
    baseline_stage = (OUTPUT / "pilot").relative_to(OUT).as_posix()
    freeze(DIRECTORY / "protocol.json", dict(
        candidate=CANDIDATE_ARM, parent=PARENT_ARM, baseline=ROUND06_QB, split_seed=2026,
        pairs=pairs_as_lists(PILOT), model=CANDIDATE, epochs=EPOCHS, batch_size=BATCH, lr=LR,
        training_eeg=SINGLE_TRIAL_SAMPLING, parent_checkpoint_stage=parent_stage,
        noise_std=NOISE_STD, parent_noise_std=TRIAL_NOISE_STD, channel_dropout=TRIAL_CHANNEL_DROPOUT,
        ema_decay=EMA_DECAY, ema_first_epoch=EMA_FIRST_EPOCH, ema_bn="last epoch buffers",
        qb_beta=BETA, qb_k=1, calibration="standard four-repeat dev_train EEG, within subject",
        gate="vs round 06 QB-Norm: Top-5 >= +1pp, Top-1 >= 0pp, at least 3/4 improve Top-5",
        new_fits=4, official_test_used=False))
    design = design_for(2026)
    indices, queries, templates, labels = (design[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels"))
    norms = np.linalg.norm(templates, axis=1)
    centers = tensor(templates)
    comparisons = {ROUND06_QB: [], PARENT_ARM: []}
    for subject, seed in PILOT:
        path = f"sub-{subject:02d}/seed-{seed}"
        baseline = run_dir(baseline_stage, CANDIDATE, subject, seed)
        parent = run_dir(parent_stage, CANDIDATE, subject, seed)
        candidate = run_dir(stage, CANDIDATE, subject, seed)
        baseline_stats = training_stats(baseline, CANDIDATE, subject, seed)
        parent_stats = training_stats(parent, CANDIDATE, subject, seed)
        reference = torch.load(baseline / "checkpoint.pt", map_location=DEVICE)
        with np.load(ROUND13 / "pilot" / path / "qb_scores.npz") as saved:
            np.testing.assert_array_equal(saved["query_rows"], queries)
            np.testing.assert_array_equal(saved["labels"], labels)
            baseline_ranks = ranks_and_accuracy(saved["scores"], labels)[0]
            np.testing.assert_array_equal(baseline_ranks, saved["ranks"])
        full = load_subject(subject)
        repeats = checked_trials(subject, indices, full)
        train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, design, split="dev", trial_repeats=repeats,
              single_trials=True, perturb_trials=True, weight_ema=True, trial_noise_std=NOISE_STD)
        del repeats
        candidate_stats = training_stats(candidate, CANDIDATE, subject, seed)
        if parent_stats["trial_noise_std"] != TRIAL_NOISE_STD:
            raise ValueError(f"Expected the original high-noise parent: {parent}")
        expected = dict(load_json(parent / "settings.json"), stage=stage, trial_noise_std=NOISE_STD)
        if load_json(candidate / "settings.json") != expected:
            raise ValueError(f"Expected only the Gaussian noise level to differ: {candidate}")
        query, train_eeg = tensor(full[queries]), tensor(full[indices])
        evaluate_checkpoint(CANDIDATE, subject, seed, design, query, centers, candidate / "ema_validation.npz",
                            stage, checkpoint_name="ema.pt")
        candidate_ranks = score_ema(candidate, candidate, design, train_eeg, query, centers, norms, reference)
        parent_ranks = score_ema(parent, DIRECTORY / "round11_control" / path, design, train_eeg, query, centers,
                                 norms, reference)
        stats = {CANDIDATE_ARM: candidate_stats, PARENT_ARM: parent_stats, ROUND06_QB: baseline_stats}
        for control, ranks in ((ROUND06_QB, baseline_ranks), (PARENT_ARM, parent_ranks)):
            row = result_row(subject, seed, CANDIDATE_ARM, control, candidate_ranks, ranks, stats)
            comparisons[control].append(row)
    summaries = {}
    for control, rows in comparisons.items():
        result = contrast(rows)
        del result["pilot_pass"], result["formal_gate_pass"]
        result["top5_subject_wins"] = top5_wins(rows)
        summaries[control] = result
    summary = dict(comparisons=summaries, gate_pass=gate("pilot", comparisons[ROUND06_QB]),
                   new_fits=4, official_test_used=False)
    write_log(DIRECTORY / "paired_runs.csv", comparisons[ROUND06_QB] + comparisons[PARENT_ARM])
    save_json(DIRECTORY / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
