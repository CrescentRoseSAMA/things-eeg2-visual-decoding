"""Compare online 2/3/4-trial EEG averages against the locked CSLS model."""
import argparse
from itertools import product

import numpy as np
import torch

from eeg_csls_replication import ADJUSTED, CANDIDATE, DIRECTORY, K, PAIRS, scoring_arms, training_stats
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import (BASE, BATCH, COURSE, EPOCHS, PILOT, contrast, design_for,
                                   freeze, pairs_as_lists, read_development, result_row)
from eeg_model import COSINE_REGRESSION_WEIGHT, PAIR_TEMPERATURE
from eeg_pooling_dev import evaluate_checkpoint
from eeg_pooling_formal import checked_metrics
from eeg_repeat_experiment import raw_trials
from eeg_repeat_views import TRIAL_SAMPLING
from eeg_train import DEVICE, LR, OUT, load_json, reload_and_score, run_dir, save_json, tensor, train, write_log


OUTPUT = BASE / "round-06"
CANDIDATE_ARM = f"{CANDIDATE}_trialaug"


def gains(rows, metric):
    return sum(row[f"top{metric}_hits"] - row[f"control_top{metric}_hits"] for row in rows)


def top5_wins(rows):
    subjects = sorted({row["subject"] for row in rows})
    return sum(gains([row for row in rows if row["subject"] == subject], 5) > 0 for subject in subjects)


def gate(stage, rows):
    pairs = PILOT if stage == "pilot" else PAIRS
    if sorted((row["subject"], row["seed"]) for row in rows) != list(pairs) or any(
            row["queries"] != 1000 for row in rows):
        raise ValueError("Expected all locked pairs on the development gallery")
    n = sum(row["queries"] for row in rows)
    if 100 * gains(rows, 5) < n:
        return False
    if stage == "pilot":
        return gains(rows, 1) >= 0 and top5_wins(rows) >= 3
    return (500 * gains(rows, 1) >= n and top5_wins(rows) >= 4
            and all(gains([row for row in rows if row["seed"] == seed], 1) >= 0 for seed in (1, 2)))


def checked_trials(subject, indices, full):
    raw = raw_trials(subject)
    values = np.empty((len(indices), 4, 17, 80), dtype=np.float32)
    for start in range(0, len(indices), 128):
        chunk = indices[start:start + 128]
        block = raw[chunk]
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite training trials for subject {subject}")
        np.testing.assert_array_equal(block.mean(1).astype(np.float32), full[chunk])
        values[start:start + len(chunk)] = block
    return values


def protocol_for(pairs):
    return dict(candidate=CANDIDATE_ARM, model=CANDIDATE, reference=ADJUSTED,
                pairs=pairs_as_lists(pairs), epochs=EPOCHS, batch_size=BATCH, learning_rate=LR,
                temperature=PAIR_TEMPERATURE, cosine_weight=COSINE_REGRESSION_WEIGHT, csls_k=K,
                training_eeg=TRIAL_SAMPLING, sampling_rng="NumPy default_rng([seed, epoch, 604])",
                preprocessing="per-electrode training moments from standard full averages",
                calibration="matched training rows, standard four-repeat average EEG",
                official_test_used=False)


def checkpoint_stats(candidate, reference, subject, seed):
    new = training_stats(candidate, CANDIDATE, subject, seed)
    old = training_stats(reference, CANDIDATE, subject, seed)
    if (new.get("training_eeg") != TRIAL_SAMPLING or "training_eeg" in old
            or new["parameters"] != old["parameters"]):
        raise ValueError("Expected a trial-only difference from the reference")
    return {CANDIDATE_ARM: new, ADJUSTED: old}


def stored_csls(path, labels):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["labels"], labels)
        scores = saved["csls_scores"]
        if scores.shape != (len(labels), 200) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid saved CSLS scores: {path}")
        ranks = ranks_and_accuracy(scores, labels)[0]
        np.testing.assert_array_equal(ranks, saved["csls_ranks"])
        return ranks


def run_development(name, pairs, split_seed):
    directory = OUTPUT / name
    stage = directory.relative_to(OUT).as_posix()
    reference_stage = (BASE / "round-05" if name == "pilot" else DIRECTORY).relative_to(OUT).as_posix()
    decision = ("Top-5 >= +1pp and Top-1 >= 0pp; at least 3/4 subjects improve Top-5" if name == "pilot" else
                "Top-5 >= +1pp and Top-1 >= +0.2pp; both seeds nonnegative Top-1; at least 4/6 improve Top-5")
    freeze(directory / "protocol.json", dict(protocol_for(pairs), split_seed=split_seed, gate=decision,
           query="standard four-repeat average, fixed 200 templates and 1000 queries"))
    design = design_for(split_seed)
    indices, templates, labels = (design[key] for key in ("dev_train", "dev_centers", "dev_labels"))
    centers = tensor(templates)
    rows = []
    last_subject = None
    repeats = None
    for subject, seed in pairs:
        full = load_subject(subject)
        if subject != last_subject:
            del repeats
            repeats = checked_trials(subject, indices, full)
            last_subject = subject
        train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, design, split="dev", trial_repeats=repeats)
        candidate = run_dir(stage, CANDIDATE, subject, seed)
        reference = run_dir(reference_stage, CANDIDATE, subject, seed)
        stats = checkpoint_stats(candidate, reference, subject, seed)
        candidate_val, reference_val = candidate / "validation.npz", reference / "validation.npz"
        query = tensor(full[design["dev_query"]])
        _, mean, std = evaluate_checkpoint(CANDIDATE, subject, seed, design, query, centers, candidate_val, stage)
        control = torch.load(reference / "checkpoint.pt", map_location=DEVICE)
        torch.testing.assert_close(mean, control["mean"], rtol=0, atol=0)
        torch.testing.assert_close(std, control["std"], rtol=0, atol=0)
        raw_ranks = read_development(candidate_val, design)
        read_development(reference_val, design)
        train_eeg = full[indices]
        with np.load(candidate_val) as saved:
            scores = scoring_arms(candidate, saved["scores"], labels, templates, train_eeg)
        np.testing.assert_array_equal(raw_ranks, scores[CANDIDATE])
        if name == "pilot":
            with np.load(reference_val) as saved:
                old = scoring_arms(reference, saved["scores"], labels, templates, train_eeg,
                                   destination=candidate / "reference_csls.npz")[ADJUSTED]
        else:
            old = stored_csls(reference / "csls.npz", labels)
        rows.append(result_row(subject, seed, CANDIDATE_ARM, ADJUSTED, scores[ADJUSTED], old, stats))
    passed = gate(name, rows)
    summarize(directory, rows, gate_pass=passed, official_test_used=False)
    return passed


def summarize(directory, rows, **status):
    result = contrast(rows)
    del result["pilot_pass"], result["formal_gate_pass"]
    result.update(top5_subject_wins=top5_wins(rows), **status)
    write_log(directory / "paired_runs.csv", rows)
    save_json(directory / "summary.json", result)
    print(directory.relative_to(OUT), result, flush=True)


def run_formal():
    directory = OUTPUT / "formal"
    stage = directory.relative_to(OUT).as_posix()
    reference_stage = (DIRECTORY / "formal").relative_to(OUT).as_posix()
    pairs = tuple(product(range(1, 11), range(5)))
    manifest = dict(protocol_for(pairs), fit="all 1654 official training classes, from scratch",
                    query="official average EEG and published 200 templates after all 50 fits")
    del manifest["official_test_used"]
    freeze(directory / "protocol.json", manifest)
    with np.load(CACHE / "design.npz") as cache:
        fit = {"train_images": cache["train_images"]}
    for subject in range(1, 11):
        if all((run_dir(stage, CANDIDATE, subject, seed) / "training.json").is_file() for seed in range(5)):
            continue
        full = load_subject(subject)
        repeats = checked_trials(subject, np.arange(len(full)), full)
        for seed in range(5):
            train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, fit, split="runs", trial_repeats=repeats)
        del repeats
    del fit
    with np.load(CACHE / "design.npz") as cache:
        test = {key: cache[key] for key in ("test_centers", "test_labels")}
    labels, templates = test["test_labels"], test["test_centers"]
    rows = []
    for subject, seed in pairs:
        reload_and_score(CANDIDATE, subject, seed, test, stage)
        candidate = run_dir(stage, CANDIDATE, subject, seed)
        reference = run_dir(reference_stage, CANDIDATE, subject, seed)
        stats = checkpoint_stats(candidate, reference, subject, seed)
        checked_metrics(candidate)
        raw = np.load(candidate / "scores.npy")
        ranks = scoring_arms(candidate, raw, labels, templates, load_subject(subject))[ADJUSTED]
        old = stored_csls(reference / "csls.npz", labels)
        rows.append(result_row(subject, seed, CANDIDATE_ARM, ADJUSTED, ranks, old, stats))
    by_seed = np.array([[np.mean([r[f"top{k}_pct"] for r in rows if r["seed"] == seed]) for k in (1, 5)]
                        for seed in range(5)])
    summarize(directory, rows, official_test_used=True, seed_sd_pp=by_seed.std(0, ddof=1).tolist(),
              delta_vs_course_pp=(by_seed.mean(0) - COURSE).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "replication", "formal"))
    stage = parser.parse_args().stage
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    if stage == "pilot":
        run_development("pilot", PILOT, 2026)
    elif stage == "replication":
        if not load_json(OUTPUT / "pilot" / "summary.json")["gate_pass"]:
            raise RuntimeError("The pilot gate did not pass")
        run_development("replication", PAIRS, 2027)
    else:
        if not load_json(OUTPUT / "replication" / "summary.json")["gate_pass"]:
            raise RuntimeError("The replication gate did not pass")
        run_formal()
