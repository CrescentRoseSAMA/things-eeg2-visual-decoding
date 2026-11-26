"""Run paired development rounds and evaluate a single locked follow-up."""
import argparse
from itertools import product

import numpy as np
import torch

from eeg_cross_subject import fresh_split
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_model import (COSINE_REGRESSION_CONFIG, COSINE_REGRESSION_WEIGHT,
                       LEARNABLE_TEMPERATURE_CONFIG, MAX_LOGIT_SCALE, PAIR_TEMPERATURE)
from eeg_pooling_dev import evaluate_checkpoint
from eeg_pooling_formal import checked_metrics
from eeg_train import (DEVICE, LR, OUT, load_json, reload_and_score, run_dir,
                       save_json, tensor, train, write_log)


BASE = OUT / "gemini_iterations"
REFERENCE = "ab_pool4"
PILOT = ((1, 0), (2, 0), (3, 1), (4, 1))
EPOCHS = 100
BATCH = 128
COURSE = (15.6, 42.8)


def freeze(path, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and load_json(path) != protocol:
        raise ValueError(f"Existing experiment protocol differs: {path}")
    save_json(path, protocol)


def design_for(split_seed):
    keys = ("train_images", "train_labels", "dev_classes", "dev_support", "dev_train",
            "dev_query", "dev_centers", "dev_labels")
    with np.load(CACHE / "design.npz") as cache:
        design = {k: cache[k] for k in keys}
    if split_seed == 2027:
        design = fresh_split(design)
    elif split_seed == 2028:
        design = fresh_split(design, 2028, exclude=fresh_split(design)["dev_classes"])
    rows, query, support = (design[k] for k in ("dev_train", "dev_query", "dev_support"))
    labels, heldout = design["train_labels"], design["dev_classes"]
    np.testing.assert_array_equal(np.sort(np.r_[rows, query, support]), np.arange(16540))
    np.testing.assert_array_equal(labels[query], heldout[design["dev_labels"]])
    np.testing.assert_array_equal(labels[support], np.repeat(heldout, 5))
    if len(rows) != 14540 or len(query) != 1000 or len(heldout) != 200 or np.intersect1d(labels[rows], heldout).size:
        raise ValueError("Unexpected development class split")
    np.testing.assert_array_equal(design["dev_centers"], design["train_images"][support.reshape(200, 5)].mean(1))
    return design


def pilot_source(config, subject, seed, directory):
    if config == REFERENCE:
        if seed == 0:
            path = OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{subject:02d}.npz"
        else:
            path = OUT / "pooling_replication" / f"ab_pool4_sub-{subject:02d}_seed-{seed}.npz"
        return path, "dev"
    for earlier in sorted(BASE.glob("round-*/protocol.json")):
        if earlier.parent == directory:
            break
        if load_json(earlier)["candidate"] == config:
            stage = earlier.parent.relative_to(OUT).as_posix()
            return run_dir(stage, config, subject, seed) / "validation.npz", stage
    raise ValueError(f"No completed earlier pilot for parent {config}")


def read_development(path, design):
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["query_rows"], design["dev_query"])
        np.testing.assert_array_equal(saved["labels"], design["dev_labels"])
        scores = saved["scores"]
        if scores.shape != (1000, 200) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid score matrix: {path}")
        ranks, _, _ = ranks_and_accuracy(scores, design["dev_labels"])
        np.testing.assert_array_equal(ranks, saved["ranks"])
    return ranks


def development(directory, protocol, pairs, split_seed):
    freeze(directory / "protocol.json", protocol)
    design = design_for(split_seed)
    controls = tuple(dict.fromkeys((REFERENCE, protocol["parent"])))
    candidate = protocol["candidate"]
    stage = directory.relative_to(OUT).as_posix()
    records = []
    for subject, seed in pairs:
        queries = tensor(load_subject(subject, "training")[design["dev_query"]])
        centers = tensor(design["dev_centers"])
        ranks, stats, moments = {}, {}, {}
        for config in (*controls, candidate):
            if split_seed == 2026 and config != candidate:
                path, source = pilot_source(config, subject, seed, directory)
                saved_ranks = read_development(path, design)
                replay = directory / f"{config}-sub-{subject:02d}-seed-{seed}-replay.npz"
            else:
                source = stage
                train(config, subject, seed, stage, EPOCHS, BATCH, design, split="dev")
                replay = run_dir(stage, config, subject, seed) / "validation.npz"
            _, mean, std = evaluate_checkpoint(config, subject, seed, design, queries, centers, replay, source)
            ranks[config] = read_development(replay, design)
            if split_seed == 2026 and config != candidate:
                np.testing.assert_array_equal(ranks[config], saved_ranks)
            moments[config] = (mean, std)
            stats[config] = load_json(run_dir(source, config, subject, seed) / "training.json")
            if (stats[config]["subject"], stats[config]["seed"], stats[config]["config"],
                stats[config]["trained_epochs"], stats[config]["batch_size"], stats[config]["lr"]) != (
                    subject, seed, config, EPOCHS, BATCH, LR):
                raise ValueError(f"Unexpected checkpoint settings: {config}, {subject}, {seed}")
        for control in controls:
            for a, b in zip(moments[candidate], moments[control], strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            records.append(result_row(subject, seed, candidate, control, ranks[candidate], ranks[control], stats))
    return save_results(directory, records)


def result_row(subject, seed, candidate, control, new, old, stats):
    data = dict(subject=subject, seed=seed, candidate=candidate, control=control, queries=len(new))
    for k in (1, 5):
        a, b = (int((r <= k).sum()) for r in (new, old))
        data.update({f"top{k}_hits": a, f"control_top{k}_hits": b,
                     f"top{k}_pct": 100 * a / len(new), f"control_top{k}_pct": 100 * b / len(new),
                     f"top{k}_delta_pp": round(100 * (a - b) / len(new), 8)})
    for key in ("parameters", "train_seconds", "peak_gpu_mib"):
        data[key], data[f"control_{key}"] = stats[candidate][key], stats[control][key]
    if "final_temperature" in stats[candidate]:
        data["final_temperature"] = stats[candidate]["final_temperature"]
        data["control_temperature"] = stats[control].get("final_temperature", stats[control]["temperature"])
    return data


def contrast(rows):
    n = sum(r["queries"] for r in rows)
    candidate = [sum(r[f"top{k}_hits"] for r in rows) for k in (1, 5)]
    control = [sum(r[f"control_top{k}_hits"] for r in rows) for k in (1, 5)]
    gains = [a - b for a, b in zip(candidate, control, strict=True)]
    subjects = sorted({r["subject"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    def top1_by(key, value):
        return sum(r["top1_hits"] - r["control_top1_hits"] for r in rows if r[key] == value)

    wins = sum(top1_by("subject", s) > 0 for s in subjects)
    return dict(runs=len(rows), candidate_pct=[100 * a / n for a in candidate],
                control_pct=[100 * b / n for b in control], delta_pp=[100 * a / n for a in gains],
                top1_subject_wins=wins, positive_top1_seeds=[s for s in seeds if top1_by("seed", s) > 0],
                parameters=rows[0]["parameters"],
                pilot_pass=1000 * gains[0] >= 5 * n and wins >= 3 and gains[1] >= 0,
                formal_gate_pass=100 * gains[0] >= n and 100 * gains[1] >= 2 * n
                and len(seeds) == 2 and all(top1_by("seed", s) > 0 for s in seeds) and wins >= 4)


def save_results(directory, rows):
    write_log(directory / "paired_runs.csv", rows)
    summary = {control: contrast([r for r in rows if r["control"] == control])
               for control in dict.fromkeys(r["control"] for r in rows)}
    save_json(directory / "summary.json", summary)
    print(directory.relative_to(OUT), summary, flush=True)
    return summary


def formal(directory, candidate):
    pairs = tuple(product(range(1, 11), range(5)))
    stage = directory.relative_to(OUT).as_posix()
    freeze(directory / "protocol.json", dict(candidate=candidate, parent=REFERENCE, pairs=pairs_as_lists(pairs),
           epochs=EPOCHS, batch=BATCH, learning_rate=LR, selection="frozen after independent development",
           fit="1654 official training categories", evaluation="200 original official categories after all 50 fits"))
    with np.load(CACHE / "design.npz") as cache:
        fit = {"train_images": cache["train_images"]}
    for subject, seed in pairs:
        train(candidate, subject, seed, stage, EPOCHS, BATCH, fit, split="runs")
    del fit
    with np.load(CACHE / "design.npz") as cache:
        test = {k: cache[k] for k in ("test_labels", "test_centers")}
    rows = []
    for subject, seed in pairs:
        reload_and_score(candidate, subject, seed, test, stage)
        ranks, stats = {}, {}
        for config, source in ((candidate, stage), (REFERENCE, "pooling_formal")):
            folder = run_dir(source, config, subject, seed)
            stats[config], _ = checked_metrics(folder)
            if (stats[config]["subject"], stats[config]["seed"], stats[config]["config"]) != (subject, seed, config):
                raise ValueError(f"Unexpected formal checkpoint settings: {folder}")
            ranks[config], _, _ = ranks_and_accuracy(np.load(folder / "scores.npy"), test["test_labels"])
        rows.append(result_row(subject, seed, candidate, REFERENCE, ranks[candidate], ranks[REFERENCE], stats))
    write_log(directory / "paired_runs.csv", rows)
    by_seed = np.array([[np.mean([r[f"top{k}_pct"] for r in rows if r["seed"] == seed])
                         for k in (1, 5)] for seed in range(5)])
    means = by_seed.mean(0)
    effect = contrast(rows)
    save_json(directory / "summary.json", dict(runs=len(rows), mean_pct=means.tolist(),
              seed_sd_pp=by_seed.std(0, ddof=1).tolist(), reference_pct=effect["control_pct"],
              delta_vs_ab_pool4_pp=effect["delta_pp"], delta_vs_course_pp=(means - COURSE).tolist()))


def pairs_as_lists(pairs):
    return [list(pair) for pair in pairs]


def finalize():
    eligible = []
    for number in range(1, 4):
        directory = BASE / f"round-{number:02d}"
        outcome = load_json(directory / "summary.json")[REFERENCE]
        if outcome["pilot_pass"]:
            candidate = load_json(directory / "protocol.json")["candidate"]
            eligible.append((outcome["delta_pp"], -outcome["parameters"], -number, candidate))
    last = BASE / "round-03"
    candidate = max(eligible)[-1] if eligible else None
    freeze(last / "final-selection" / "protocol.json", dict(candidate=candidate,
           eligible=[entry[-1] for entry in eligible], selection="Top-1, Top-5, then fewer parameters"))
    formal_used = False
    if candidate:
        pairs = tuple(product(range(5, 11), (1, 2)))
        heldout = last / "replication"
        outcome = development(heldout, dict(candidate=candidate, parent=REFERENCE, split_seed=2027,
                              pairs=pairs_as_lists(pairs), epochs=EPOCHS, batch=BATCH, learning_rate=LR), pairs, 2027)
        formal_used = outcome[REFERENCE]["formal_gate_pass"]
        if formal_used:
            formal(last / "formal", candidate)
    save_json(last / "final-selection" / "summary.json", dict(candidate=candidate, formal_test_used=formal_used))
    print(f"Final selection: {candidate}; official test used: {formal_used}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", choices=("1", "2", "3", "4", "5", "finalize"))
    parser.add_argument("--candidate")
    parser.add_argument("--parent", default=REFERENCE)
    parser.add_argument("--hypothesis")
    args = parser.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    if args.round == "finalize":
        finalize()
        return
    if not args.candidate or not args.hypothesis or args.candidate in {REFERENCE, args.parent}:
        parser.error("a distinct --candidate and a --hypothesis are required")
    directory = BASE / f"round-{int(args.round):02d}"
    protocol = dict(candidate=args.candidate, parent=args.parent, hypothesis=args.hypothesis,
                    reference=REFERENCE, pairs=pairs_as_lists(PILOT), epochs=EPOCHS, batch=BATCH,
                    learning_rate=LR, split_seed=2026, loss="symmetric EEG-image InfoNCE",
                    scoring="epoch 100, unit EEG embedding dot frozen CLIP template",
                    normalization="per electrode, training rows only", official_test_used=False)
    if args.candidate == LEARNABLE_TEMPERATURE_CONFIG:
        protocol.update(initial_temperature=PAIR_TEMPERATURE, max_logit_scale=MAX_LOGIT_SCALE,
                        temperature="learned scalar in training loss only")
    if args.candidate == COSINE_REGRESSION_CONFIG:
        protocol.update(loss="weighted symmetric InfoNCE and matched-pair cosine distance",
                        cosine_regression_weight=COSINE_REGRESSION_WEIGHT, temperature=PAIR_TEMPERATURE)
    development(directory, protocol, PILOT, 2026)


if __name__ == "__main__":
    main()
