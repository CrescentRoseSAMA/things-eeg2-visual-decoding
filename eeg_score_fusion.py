"""Evaluate two fixed CSLS score fusions on a fresh development gallery."""
import numpy as np
import torch

from eeg_cross_subject import SUBJECTS
from eeg_csls_replication import ADJUSTED, CANDIDATE, K, PAIRS, scoring_arms, training_stats
from eeg_data import load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import (BASE, BATCH, EPOCHS, contrast, design_for, freeze, pairs_as_lists, read_development,
                                   result_row)
from eeg_model import TRANSFER_CONFIG
from eeg_pooling_dev import evaluate_checkpoint
from eeg_repeat_views import TRIAL_SAMPLING
from eeg_train import DEVICE, LR, OUT, run_dir, save_json, save_weights, tensor, train, write_log
from eeg_transfer_pilot import FINETUNE_EPOCHS, FINETUNE_LR, PRETRAIN_EPOCHS, pretrain, target_state
from eeg_trial_augmentation import checked_trials, gains, top5_wins


DIRECTORY = BASE / "round-09" / "development"
SINGLE = "round06_single"
TWO_SEEDS = "round06_two_seeds"
MIXED = "round06_round08_fusion"
COMPARISONS = ((TWO_SEEDS, SINGLE), (MIXED, SINGLE), (MIXED, TWO_SEEDS))


def fusion_gate(rows):
    if sorted((row["subject"], row["seed"]) for row in rows) != list(PAIRS) or any(
            row["queries"] != 1000 for row in rows):
        raise ValueError("Expected all locked development pairs for the fusion comparison")
    queries = sum(row["queries"] for row in rows)
    return 200 * gains(rows, 5) >= 3 * queries and gains(rows, 1) >= 0 and top5_wins(rows) >= 4


def fit(config, subject, seed, stage, design, full, repeats, centers, initial):
    epochs, lr = (EPOCHS, LR) if initial is None else (FINETUNE_EPOCHS, FINETUNE_LR)
    train(config, subject, seed, stage, epochs, BATCH, design, split="dev", trial_repeats=repeats,
          initial_checkpoint=initial, lr=lr)
    folder = run_dir(stage, config, subject, seed)
    validation = folder / "validation.npz"
    _, mean, std = evaluate_checkpoint(config, subject, seed, design, tensor(full[design["dev_query"]]), centers,
                                       validation, stage, epochs=epochs)
    read_development(validation, design)
    labels = design["dev_labels"]
    with np.load(validation) as raw:
        ranks = scoring_arms(folder, raw["scores"], labels, design["dev_centers"], full[design["dev_train"]],
                             model_config=config)[ADJUSTED]
    with np.load(folder / "csls.npz") as saved:
        np.testing.assert_array_equal(saved["labels"], labels)
        np.testing.assert_array_equal(saved["csls_ranks"], ranks)
        scores = saved["csls_scores"]
    stats = training_stats(folder, config, subject, seed, epochs=epochs, lr=lr)
    if stats.get("training_eeg") != TRIAL_SAMPLING or stats["parameters"] != (889353 if initial is None else 889642):
        raise ValueError(f"Unexpected model training settings: {folder}")
    return scores, ranks, stats, (mean, std)


def fuse(folder, first, second, design):
    scores = (first[0] + second[0]) / 2
    labels = design["dev_labels"]
    ranks = ranks_and_accuracy(scores, labels)[0]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "fusion.npz", scores=scores, labels=labels, ranks=ranks,
                        query_rows=design["dev_query"])
    a, b = first[2], second[2]
    return ranks, dict(parameters=a["parameters"] + b["parameters"],
                       train_seconds=a["train_seconds"] + b["train_seconds"],
                       peak_gpu_mib=max(a["peak_gpu_mib"], b["peak_gpu_mib"]))


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    design = design_for(2028)
    freeze(DIRECTORY / "protocol.json", dict(
        split_seed=2028, excluded_validation_splits=[2026, 2027], validation_classes=design["dev_classes"].tolist(),
        training_classes=1454, pairs=pairs_as_lists(PAIRS), batch_size=BATCH, training_eeg=TRIAL_SAMPLING,
        round06_epochs=EPOCHS, round06_lr=LR, round08_pretrain_subjects=list(SUBJECTS),
        round08_pretrain_seeds=[1, 2], round08_pretrain_epochs=PRETRAIN_EPOCHS, round08_pretrain_lr=LR,
        round08_finetune_epochs=FINETUNE_EPOCHS, round08_finetune_lr=FINETUNE_LR,
        csls_k=K, query="1000 standard EEG queries with 200 fixed CLIP templates",
        calibration="each model uses the target's standard four-repeat training averages",
        fusion="equal-weight mean of each model's CSLS scores before ranking",
        same_architecture="fuse seeds 1 and 2 once per subject, pair against each corresponding single-model row",
        mixed="fuse the round 06 and round 08 models with the same target subject and seed",
        comparisons=pairs_as_lists(COMPARISONS),
        gate="versus round06_single: Top-5 >= +1.5pp and Top-1 >= 0pp, at least 4/6 subjects improve Top-5",
        official_test_used=False))
    pretrain_dirs = {seed: DIRECTORY / "pretrain" / f"seed-{seed}" for seed in (1, 2)}
    shared = {}
    for seed, folder in pretrain_dirs.items():
        shared[seed] = pretrain(folder, design, seed)
        save_json(folder / "summary.json", shared[seed]["stats"])
    centers = tensor(design["dev_centers"])
    stage = DIRECTORY.relative_to(OUT).as_posix()
    rows = []
    single_seconds = transfer_seconds = 0.0
    for subject in sorted({subject for subject, _ in PAIRS}):
        full = load_subject(subject)
        repeats = checked_trials(subject, design["dev_train"], full)
        singles, mixed = {}, {}
        subject_dir = DIRECTORY / "fusions" / f"sub-{subject:02d}"
        for seed in (1, 2):
            single = fit(CANDIDATE, subject, seed, stage, design, full, repeats, centers, None)
            initial = pretrain_dirs[seed] / f"target-{subject:02d}.pt"
            save_weights(initial, dict(model=target_state(shared[seed]["model"], SUBJECTS.index(subject))))
            transfer = fit(TRANSFER_CONFIG, subject, seed, stage, design, full, repeats, centers, initial)
            torch.testing.assert_close(single[3], transfer[3], rtol=0, atol=0)
            singles[seed] = single
            mixed[seed] = fuse(subject_dir / f"mixed-seed-{seed}", single, transfer, design)
            single_seconds += single[2]["train_seconds"]
            transfer_seconds += transfer[2]["train_seconds"]
        two_seeds = fuse(subject_dir / "round06-two-seeds", singles[1], singles[2], design)
        for seed in (1, 2):
            results = {SINGLE: (singles[seed][1], singles[seed][2]), TWO_SEEDS: two_seeds, MIXED: mixed[seed]}
            stats = {name: result[1] for name, result in results.items()}
            for candidate, control in COMPARISONS:
                rows.append(result_row(subject, seed, candidate, control, results[candidate][0],
                                       results[control][0], stats))
    summaries = {}
    for candidate, control in COMPARISONS:
        paired = [r for r in rows if (r["candidate"], r["control"]) == (candidate, control)]
        summary = contrast(paired)
        del summary["pilot_pass"], summary["formal_gate_pass"]
        summary["top5_subject_wins"] = top5_wins(paired)
        if control == SINGLE:
            summary["gate_pass"] = fusion_gate(paired)
        summaries[f"{candidate}_vs_{control}"] = summary
    pretrain_seconds = sum(state["stats"]["train_seconds"] for state in shared.values())
    write_log(DIRECTORY / "paired_runs.csv", rows)
    summary = dict(comparisons=summaries, round06_training_seconds=single_seconds,
                   round08_pretrain_seconds=pretrain_seconds, round08_finetuning_seconds=transfer_seconds,
                   total_training_seconds=single_seconds + pretrain_seconds + transfer_seconds,
                   official_test_used=False)
    save_json(DIRECTORY / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
