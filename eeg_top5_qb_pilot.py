"""Evaluate a Top-5 training loss with fixed QB-Norm."""

import argparse
from itertools import product

import numpy as np
import torch

from eeg_csls_replication import CANDIDATE, training_stats
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import BASE, BATCH, COURSE, EPOCHS, PILOT, design_for, freeze, pairs_as_lists, read_development, result_row
from eeg_pooling_dev import evaluate_checkpoint
from eeg_pooling_formal import checked_metrics
from eeg_qb_csls_pilot import CANDIDATE_ARM as ROUND06_QB, DIRECTORY as ROUND13
from eeg_querybank import BETA, qb_scores, querybank_statistics
from eeg_train import (DEVICE, LR, OUT, TOP5_GALLERY_SIZE, TOP5_RANK_WEIGHT, load_json, model_for,
                       reload_and_score, run_dir, score, tensor, train)
from eeg_trial_augmentation import OUTPUT, checked_trials, gate, summarize


DIRECTORY = BASE / "round-15" / "pilot"
CANDIDATE_ARM = "round06_top5_margin_qb"


def read_baseline(phase, subject, seed, queries, labels):
    path = ROUND13 / phase / f"sub-{subject:02d}" / f"seed-{seed}" / "qb_scores.npz"
    with np.load(path) as old:
        np.testing.assert_array_equal(old["query_rows"], queries)
        np.testing.assert_array_equal(old["labels"], labels)
        ranks = ranks_and_accuracy(old["scores"], labels)[0]
        np.testing.assert_array_equal(ranks, old["ranks"])
    return ranks


def check_settings(candidate, baseline, stage):
    expected = dict(load_json(baseline / "settings.json"), stage=stage,
                    top5_rank_weight=TOP5_RANK_WEIGHT, top5_gallery_size=TOP5_GALLERY_SIZE)
    if load_json(candidate / "settings.json") != expected:
        raise ValueError(f"Expected only the Top-5 auxiliary loss to differ: {candidate}")


def score_with_qb(candidate, baseline, raw, train_eeg, centers, norms, indices, queries, labels):
    reference = torch.load(baseline / "checkpoint.pt", map_location=DEVICE)
    checkpoint = torch.load(candidate / "checkpoint.pt", map_location=DEVICE)
    mean, std = checkpoint["mean"], checkpoint["std"]
    torch.testing.assert_close((mean, std), (reference["mean"], reference["std"]), rtol=0, atol=0)
    model = model_for(CANDIDATE)
    model.load_state_dict(checkpoint["model"], strict=True)
    bank = score(model, (tensor(train_eeg) - mean) / std, centers, BATCH).cpu().numpy() / norms
    cosine = raw / norms
    if bank.shape != (len(indices), len(norms)) or not np.isfinite(bank).all():
        raise ValueError(f"Invalid QB-Norm training bank: {candidate}")
    stats = querybank_statistics(bank)
    adjusted, _ = qb_scores(cosine, stats)
    ranks = ranks_and_accuracy(adjusted, labels)[0]
    np.savez_compressed(candidate / "qb_scores.npz", query_rows=queries, train_rows=indices, labels=labels,
                        scores=adjusted, ranks=ranks, bank_cosine=bank, normalized_scores=cosine, **stats)
    return ranks


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    stage = DIRECTORY.relative_to(OUT).as_posix()
    baseline_stage = (OUTPUT / "pilot").relative_to(OUT).as_posix()
    freeze(DIRECTORY / "protocol.json", dict(
        candidate=CANDIDATE_ARM, baseline=ROUND06_QB, split_seed=2026,
        pairs=pairs_as_lists(PILOT), model=CANDIDATE, epochs=EPOCHS, batch_size=BATCH, lr=LR,
        auxiliary_loss="relu((fifth highest incorrect cosine - correct cosine) / 0.07)",
        top5_rank_weight=TOP5_RANK_WEIGHT, gallery_size=TOP5_GALLERY_SIZE,
        templates="unit-normalized class means of fixed training image features; positive excludes paired image",
        gallery="200 drawn training classes per batch; each query uses first 199 draws excluding its class",
        gallery_rng="independent torch generator, reseeded with seed * 2**32 + epoch",
        qb_beta=BETA, qb_k=1, calibration="standard four-repeat dev_train EEG, within subject",
        gate="vs round 06 QB-Norm: Top-5 >= +1pp, Top-1 >= 0pp, at least 3/4 improve Top-5",
        new_fits=4, official_test_used=False))
    design = design_for(2026)
    indices, queries, templates, labels = (design[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels"))
    train_classes = design["train_labels"][indices]
    if np.intersect1d(train_classes, design["train_labels"][queries]).size:
        raise ValueError("Expected training and development query classes to be disjoint")
    norms = np.linalg.norm(templates, axis=1)
    centers = tensor(templates)
    rows = []
    for subject, seed in PILOT:
        baseline = run_dir(baseline_stage, CANDIDATE, subject, seed)
        candidate = run_dir(stage, CANDIDATE, subject, seed)
        base_stats = training_stats(baseline, CANDIDATE, subject, seed)
        control = read_baseline("pilot", subject, seed, queries, labels)
        full = load_subject(subject)
        repeats = checked_trials(subject, indices, full)
        train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, design, split="dev", trial_repeats=repeats,
              class_labels=train_classes, top5_ranking=True)
        del repeats
        candidate_stats = training_stats(candidate, CANDIDATE, subject, seed)
        check_settings(candidate, baseline, stage)
        validation = candidate / "validation.npz"
        evaluate_checkpoint(CANDIDATE, subject, seed, design, tensor(full[queries]), centers, validation, stage)
        read_development(validation, design)
        with np.load(validation) as saved:
            raw = saved["scores"]
        ranks = score_with_qb(candidate, baseline, raw, full[indices], centers, norms, indices, queries, labels)
        rows.append(result_row(subject, seed, CANDIDATE_ARM, ROUND06_QB, ranks, control,
                               {CANDIDATE_ARM: candidate_stats, ROUND06_QB: base_stats}))
    summarize(DIRECTORY, rows, gate_pass=gate("pilot", rows), new_fits=4, official_test_used=False)


def run_full():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    directory = DIRECTORY.parent / "full"
    stage = directory.relative_to(OUT).as_posix()
    baseline_stage = (OUTPUT / "formal").relative_to(OUT).as_posix()
    subjects, seeds = range(1, 11), range(5)
    pairs = tuple(product(subjects, seeds))
    pilot = load_json(DIRECTORY / "summary.json")
    protocol = load_json(DIRECTORY / "protocol.json")
    if (protocol["top5_rank_weight"], protocol["gallery_size"], protocol["qb_beta"], protocol["qb_k"]) != (
            TOP5_RANK_WEIGHT, TOP5_GALLERY_SIZE, BETA, 1):
        raise ValueError("Full evaluation requires the locked pilot settings")
    del protocol["split_seed"], protocol["gate"]
    protocol.update(pairs=pairs_as_lists(pairs), fit="all 1654 official training classes, from scratch",
                    calibration="16540 standard four-repeat official training EEG averages, within subject",
                    query="200 official eighty-repeat EEG averages per subject and seed after all 50 fits",
                    selection="user-requested full evaluation using the fixed pilot configuration",
                    pilot_gate_pass=pilot["gate_pass"], new_fits=len(pairs), official_test_used=True,
                    course_target_pct=list(COURSE))
    freeze(directory / "protocol.json", protocol)
    with np.load(CACHE / "design.npz") as cache:
        fit = {"train_images": cache["train_images"]}
        train_classes = cache["train_labels"]
    if len(train_classes) != 16540 or len(np.unique(train_classes)) != 1654:
        raise ValueError("Expected the complete official training split")
    indices = np.arange(len(train_classes))
    for subject in subjects:
        if not all((run_dir(stage, CANDIDATE, subject, seed) / "training.json").is_file() for seed in seeds):
            repeats = checked_trials(subject, indices, load_subject(subject))
            for seed in seeds:
                train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, fit, split="runs", trial_repeats=repeats,
                      class_labels=train_classes, top5_ranking=True)
            del repeats
        for seed in seeds:
            candidate = run_dir(stage, CANDIDATE, subject, seed)
            baseline = run_dir(baseline_stage, CANDIDATE, subject, seed)
            check_settings(candidate, baseline, stage)
            training_stats(candidate, CANDIDATE, subject, seed)
        print(f"Full training complete for sub-{subject:02d} (5 seeds)", flush=True)
    del fit

    with np.load(CACHE / "design.npz") as cache:
        test = {key: cache[key] for key in ("test_centers", "test_labels")}
    templates, labels = test["test_centers"], test["test_labels"]
    queries = np.arange(200)
    np.testing.assert_array_equal(labels, queries)
    norms = np.linalg.norm(templates, axis=1)
    centers = tensor(templates)
    rows = []
    for subject in subjects:
        train_eeg = load_subject(subject)
        for seed in seeds:
            candidate = run_dir(stage, CANDIDATE, subject, seed)
            baseline = run_dir(baseline_stage, CANDIDATE, subject, seed)
            base_stats = training_stats(baseline, CANDIDATE, subject, seed)
            candidate_stats = training_stats(candidate, CANDIDATE, subject, seed)
            control = read_baseline("full", subject, seed, queries, labels)
            reload_and_score(CANDIDATE, subject, seed, test, stage)
            checked_metrics(candidate)
            raw = np.load(candidate / "scores.npy")
            ranks = score_with_qb(candidate, baseline, raw, train_eeg, centers, norms, indices, queries, labels)
            rows.append(result_row(subject, seed, CANDIDATE_ARM, ROUND06_QB, ranks, control,
                                   {CANDIDATE_ARM: candidate_stats, ROUND06_QB: base_stats}))
    by_seed = np.array([[np.mean([row[f"top{k}_pct"] for row in rows if row["seed"] == seed]) for k in (1, 5)]
                        for seed in seeds])
    summarize(directory, rows, new_fits=len(pairs), official_test_used=True, pilot_gate_pass=pilot["gate_pass"],
              seed_sd_pp=by_seed.std(0, ddof=1).tolist(), delta_vs_course_pp=(by_seed.mean(0) - COURSE).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="train and evaluate all ten subjects and five seeds")
    if parser.parse_args().full:
        run_full()
    else:
        main()
