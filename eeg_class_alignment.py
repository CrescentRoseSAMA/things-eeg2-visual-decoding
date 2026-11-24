"""Gated comparison of leave-one-out class alignment with the matched pooling control."""
import numpy as np

from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_pooling_dev import evaluate_checkpoint
from eeg_train import (DEVICE, OUT, load_json, reload_and_score, run_dir,
                       save_json, tensor, train, write_log)


MODEL = "ab_pool4"
BASE = OUT / "class_alignment"
PROTOCOL = dict(model=MODEL, control="ab_pool4 with symmetric image InfoNCE",
                optimizer="Adam", learning_rate=0.0002, epochs=100, batch=128, temperature=0.07,
                loss="0.75 image InfoNCE + 0.25 training class CE; own mean excludes paired image",
                centers="mean of unit training CLIP features, without renormalizing class means",
                pilot="subjects 1,2 seed 0; both Top1 gains > 0 and mean Top5 gain >= 0",
                replication="subjects 3..10 seeds 1,2; each seed mean Top1 gain > 0, >=5/8 subjects "
                            "with positive seed-averaged Top1 gain, mean Top1 gain >=0.5pp and Top5 gain >=0",
                formal="only after both gates pass; all 10 subjects x seeds 0..4 fit before official test",
                selection="fixed epoch 100; paired identical queries; reused official test set")


def paired(subject, seed, folder, rows, labels, new, old):
    new_rank, _, _ = ranks_and_accuracy(new, labels)
    old_rank, _, _ = ranks_and_accuracy(old, labels)
    write_log(folder / "paired_queries.csv", [
        dict(query_row=int(idx), label=int(label), control_rank=int(a), candidate_rank=int(b),
             rank_gain=int(a-b)) for idx, label, a, b in zip(rows, labels, old_rank, new_rank)])
    result = dict(subject=subject, seed=seed, queries=len(rows))
    for k in (1, 5):
        a, b = int((old_rank <= k).sum()), int((new_rank <= k).sum())
        result.update({f"control_top{k}_hits": a, f"top{k}_hits": b, f"gain{k}": b-a})
    return result


def gate(name, runs):
    if name == "pilot":
        return all(r["gain1"] > 0 for r in runs) and sum(r["gain5"] for r in runs) >= 0
    return (all(sum(r["gain1"] for r in runs if r["seed"] == seed) > 0 for seed in (1, 2))
            and sum(sum(r["gain1"] for r in runs if r["subject"] == sub) > 0 for sub in range(3, 11)) >= 5
            and sum(r["gain1"] for r in runs) >= 5 * len(runs)
            and sum(r["gain5"] for r in runs) >= 0)


def summarize(name, runs, base=BASE):
    folder = base / name
    write_log(folder / "paired_runs.csv", runs)
    n = sum(r["queries"] for r in runs)
    summary = dict(stage=name, runs=len(runs), query_evaluations=n)
    for k in (1, 5):
        for key, column in (("candidate", f"top{k}_hits"), ("control", f"control_top{k}_hits"),
                            ("delta", f"gain{k}")):
            unit = "pp" if key == "delta" else "pct"
            summary[f"{key}_top{k}_{unit}"] = 100 * sum(r[column] for r in runs) / n
    if name != "formal":
        summary["passed"] = gate(name, runs)
    save_json(folder / "summary.json", summary)
    print(summary, flush=True)
    return summary


def development(name, subjects, seeds, design, base=BASE, repeat_loader=None, *, config=MODEL, class_loss=True):
    stage = base.name + "/" + name
    classes = design["train_labels"][design["dev_train"]]
    heldout = design["train_labels"][design["dev_query"]]
    if (len(np.unique(classes)) != 1454 or len(np.unique(heldout)) != 200
            or np.intersect1d(classes, heldout).size or len(heldout) != 1000):
        raise ValueError("Unexpected development split")
    runs = []
    for sub in subjects:
        training_input = dict(class_labels=classes) if class_loss else {}
        if repeat_loader is not None:
            training_input = dict(repeat_views=repeat_loader(sub, design["dev_train"]))
        for seed in seeds:
            control = (OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{sub:02d}.npz"
                       if name == "pilot" else OUT / "pooling_replication" / f"ab_pool4_sub-{sub:02d}_seed-{seed}.npz")
            rows, labels = design["dev_query"], design["dev_labels"]
            with np.load(control) as cached:
                np.testing.assert_array_equal(cached["query_rows"], rows)
                np.testing.assert_array_equal(cached["labels"], labels)
                old = cached["scores"]
            if old.shape != (1000, 200):
                raise ValueError("Unexpected control score matrix")
            train(config, sub, seed, stage, 100, 128, design, split="dev", **training_input)
            folder = run_dir(stage, config, sub, seed)
            eeg = tensor(load_subject(sub, "training")[rows])
            path = folder / "validation.npz"
            evaluate_checkpoint(config, sub, seed, design, eeg, tensor(design["dev_centers"]), path, stage)
            with np.load(path) as candidate:
                runs.append(paired(sub, seed, folder, rows, labels, candidate["scores"], old))
    return summarize(name, runs, base)


def formal(design):
    stage = "class_alignment/formal"
    pairs = [(s, t) for s in range(1, 11) for t in range(5)]
    for sub, seed in pairs:
        train(MODEL, sub, seed, stage, 100, 128, design, split="runs", class_labels=design["train_labels"])
    with np.load(CACHE / "design.npz") as data:
        testing = {k: data[k] for k in ("test_centers", "test_labels")}
    runs = []
    for sub, seed in pairs:
        reload_and_score(MODEL, sub, seed, testing, stage)
        folder = run_dir(stage, MODEL, sub, seed)
        old = np.load(run_dir("pooling_formal", MODEL, sub, seed) / "scores.npy")
        runs.append(paired(sub, seed, folder, np.arange(200), testing["test_labels"],
                           np.load(folder / "scores.npy"), old))
    return summarize("formal", runs)


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Saved protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as data:
        design = {k: data[k] for k in ("train_images", "train_labels", "dev_train", "dev_query",
                                        "dev_labels", "dev_centers")}
    stages = []
    for name, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        stages.append(development(name, subjects, seeds, design))
        if not stages[-1]["passed"]:
            save_json(BASE / "summary.json", dict(stopped_at=name, formal_test_used=False, stages=stages))
            return
    stages.append(formal(design))
    save_json(BASE / "summary.json", dict(stopped_at="formal", formal_test_used=True, stages=stages))


if __name__ == "__main__":
    main()
