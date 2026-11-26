"""Replicate training-bank CSLS on a locked class split and gate formal scoring."""
from itertools import product

import numpy as np
import torch

from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import (BASE, BATCH, COURSE, EPOCHS, REFERENCE, contrast,
                                   design_for, freeze, pairs_as_lists, read_development, result_row)
from eeg_model import COSINE_REGRESSION_CONFIG, COSINE_REGRESSION_WEIGHT, PAIR_TEMPERATURE
from eeg_pooling_dev import evaluate_checkpoint
from eeg_pooling_formal import checked_metrics
from eeg_train import (DEVICE, LR, OUT, load_json, model_for, read_log, reload_and_score,
                       run_dir, save_json, score, tensor, train, write_log)


DIRECTORY = BASE / "round-05" / "csls_replication"
CANDIDATE = COSINE_REGRESSION_CONFIG
NORMALIZED = "round05_cosine_normalized"
ADJUSTED = "round05_cosine_csls"
PAIRS = tuple(product(range(5, 11), (1, 2)))
K = 10
COMPARISONS = ((CANDIDATE, REFERENCE), (NORMALIZED, CANDIDATE), (ADJUSTED, NORMALIZED),
               (ADJUSTED, REFERENCE))


def csls_scores(query, bank, k=K):
    """Adjust cosine scores using neighbors from the gallery and training EEG only."""
    template_neighbors = np.partition(bank, -k, axis=0)[-k:].mean(axis=0)
    query_neighbors = np.partition(query, -k, axis=1)[:, -k:].mean(axis=1)
    return 2 * query - query_neighbors[:, None] - template_neighbors


def formal_gate(rows):
    comparisons = {control: [r for r in rows if r["candidate"] == ADJUSTED and r["control"] == control]
                   for control in (REFERENCE, NORMALIZED)}
    for subset in comparisons.values():
        if sorted((r["subject"], r["seed"]) for r in subset) != list(PAIRS) or any(
                r["queries"] != 1000 for r in subset):
            raise ValueError("Formal selection requires all locked development pairs")
    return (contrast(comparisons[REFERENCE])["formal_gate_pass"]
            and sum(r["top1_hits"] - r["control_top1_hits"] for r in comparisons[NORMALIZED]) > 0)


def training_stats(folder, config, subject, seed):
    stats = load_json(folder / "training.json")
    expected = (config, subject, seed, EPOCHS, BATCH, LR, PAIR_TEMPERATURE)
    actual = tuple(stats[k] for k in ("config", "subject", "seed", "trained_epochs", "batch_size",
                                      "lr", "temperature"))
    if actual != expected or [int(row["epoch"]) for row in read_log(folder / "train.csv")] != list(
            range(1, EPOCHS + 1)):
        raise ValueError(f"Unexpected checkpoint training settings: {folder}")
    if config == CANDIDATE and stats.get("cosine_regression_weight") != COSINE_REGRESSION_WEIGHT:
        raise ValueError(f"Unexpected cosine regression weight: {folder}")
    return stats


def scoring_arms(folder, raw, labels, templates, training_eeg, *, destination=None):
    """Compare three scores for one fixed checkpoint and persist sample-level evidence."""
    ckpt = torch.load(folder / "checkpoint.pt", map_location=DEVICE)
    model = model_for(CANDIDATE)
    model.load_state_dict(ckpt["model"], strict=True)
    inputs = (tensor(training_eeg) - ckpt["mean"]) / ckpt["std"]
    bank = score(model, inputs, tensor(templates), BATCH).cpu().numpy()
    norms = np.linalg.norm(templates, axis=1)
    if (bank.shape != (len(training_eeg), 200) or raw.shape != (len(labels), 200)
            or not np.isfinite(bank).all() or not np.isfinite(raw).all() or not np.all(norms > 0)):
        raise ValueError("Invalid fixed-gallery scores or training bank")
    cosine = raw / norms
    adjusted = csls_scores(cosine, bank / norms)
    scores = {CANDIDATE: raw, NORMALIZED: cosine, ADJUSTED: adjusted}
    ranks = {name: ranks_and_accuracy(values, labels)[0] for name, values in scores.items()}
    destination = folder / "csls.npz" if destination is None else destination
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, labels=labels, normalized_scores=cosine, csls_scores=adjusted,
             normalized_ranks=ranks[NORMALIZED], csls_ranks=ranks[ADJUSTED])
    temporary.replace(destination)
    return ranks


def result_rows(subject, seed, ranks, stats):
    return [result_row(subject, seed, candidate, control, ranks[candidate], ranks[control], stats)
            for candidate, control in COMPARISONS]


def get_comparisons(rows):
    return {f"{candidate}_vs_{control}": contrast(
        [r for r in rows if (r["candidate"], r["control"]) == (candidate, control)])
            for candidate, control in COMPARISONS}


def run_replication():
    protocol = dict(candidate=CANDIDATE, reference=REFERENCE, scoring_comparisons=pairs_as_lists(COMPARISONS),
                    pairs=pairs_as_lists(PAIRS), split_seed=2027, epochs=EPOCHS, batch_size=BATCH,
                    learning_rate=LR, temperature=PAIR_TEMPERATURE, cosine_weight=COSINE_REGRESSION_WEIGHT,
                    initialization="train both encoders from scratch for each pair",
                    scoring="epoch 100; normalized template cosine and CSLS use identical round-05 checkpoint",
                    csls_k=K, calibration="14540 training EEG rows per corresponding subject and split",
                    gate="CSLS vs ab_pool4: mean Top-1 >= +1pp, mean Top-5 >= +2pp, both seeds positive Top-1, "
                         "at least four positive subjects; CSLS mean Top-1 > normalization only",
                    official_test_used=False)
    freeze(DIRECTORY / "protocol.json", protocol)
    design = design_for(2027)
    stage = DIRECTORY.relative_to(OUT).as_posix()
    templates = design["dev_centers"]
    centers = tensor(templates)
    rows = []
    for subject, seed in PAIRS:
        eeg = load_subject(subject)
        queries = tensor(eeg[design["dev_query"]])
        stats, ranks, moments = {}, {}, {}
        for config in (REFERENCE, CANDIDATE):
            train(config, subject, seed, stage, EPOCHS, BATCH, design, split="dev")
            folder = run_dir(stage, config, subject, seed)
            path = folder / "validation.npz"
            _, mean, std = evaluate_checkpoint(config, subject, seed, design, queries, centers, path, stage)
            ranks[config] = read_development(path, design)
            moments[config] = (mean, std)
            stats[config] = training_stats(folder, config, subject, seed)
        for a, b in zip(moments[REFERENCE], moments[CANDIDATE], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with np.load(path) as validation:
            adjusted = scoring_arms(folder, validation["scores"], design["dev_labels"], templates,
                                    eeg[design["dev_train"]])
        np.testing.assert_array_equal(adjusted[CANDIDATE], ranks[CANDIDATE])
        ranks.update(adjusted)
        stats[NORMALIZED] = stats[ADJUSTED] = stats[CANDIDATE]
        rows.extend(result_rows(subject, seed, ranks, stats))
        write_log(DIRECTORY / "paired_runs.csv", rows)
    passed = formal_gate(rows)
    summary = dict(pairs=len(PAIRS), training_runs=2 * len(PAIRS), official_test_used=False,
                   csls_k=K, comparisons=get_comparisons(rows), formal_gate_pass=passed)
    save_json(DIRECTORY / "summary.json", summary)
    print(summary, flush=True)
    return passed


def run_formal():
    directory = DIRECTORY / "formal"
    stage = directory.relative_to(OUT).as_posix()
    pairs = tuple(product(range(1, 11), range(5)))
    freeze(directory / "protocol.json", dict(candidate=CANDIDATE, reference=REFERENCE, pairs=pairs_as_lists(pairs),
           fit="all 1654 official training categories; from scratch", epochs=EPOCHS, batch_size=BATCH,
           learning_rate=LR, temperature=PAIR_TEMPERATURE, cosine_weight=COSINE_REGRESSION_WEIGHT,
           scoring="published 200 test templates after all 50 fits; original reference scores paired by subject and seed",
           csls_k=K, calibration="16540 official training EEG rows per corresponding subject"))
    with np.load(CACHE / "design.npz") as cache:
        fit = {"train_images": cache["train_images"]}
    for subject, seed in pairs:
        train(CANDIDATE, subject, seed, stage, EPOCHS, BATCH, fit, split="runs")
    del fit
    with np.load(CACHE / "design.npz") as cache:
        test = {k: cache[k] for k in ("test_centers", "test_labels")}
    labels, templates = test["test_labels"], test["test_centers"]
    rows = []
    for subject, seed in pairs:
        reload_and_score(CANDIDATE, subject, seed, test, stage)
        folder = run_dir(stage, CANDIDATE, subject, seed)
        reference = run_dir("pooling_formal", REFERENCE, subject, seed)
        stats = {CANDIDATE: training_stats(folder, CANDIDATE, subject, seed),
                 REFERENCE: training_stats(reference, REFERENCE, subject, seed)}
        checked_metrics(folder)
        checked_metrics(reference)
        ranks = scoring_arms(folder, np.load(folder / "scores.npy"), labels, templates, load_subject(subject))
        ranks[REFERENCE] = ranks_and_accuracy(np.load(reference / "scores.npy"), labels)[0]
        stats[NORMALIZED] = stats[ADJUSTED] = stats[CANDIDATE]
        rows.extend(result_rows(subject, seed, ranks, stats))
    write_log(directory / "paired_runs.csv", rows)
    main = [r for r in rows if r["candidate"] == ADJUSTED and r["control"] == REFERENCE]
    by_seed = np.array([[np.mean([r[f"top{k}_pct"] for r in main if r["seed"] == seed]) for k in (1, 5)]
                        for seed in range(5)])
    mean = by_seed.mean(0)
    summary = dict(runs=len(pairs), official_test_used=True, csls_k=K, mean_pct=mean.tolist(),
                   seed_sd_pp=by_seed.std(0, ddof=1).tolist(), delta_vs_course_pp=(mean - COURSE).tolist(),
                   comparisons=get_comparisons(rows))
    save_json(directory / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    if run_replication():
        print("Independent replication passed; starting formal training.", flush=True)
        run_formal()
