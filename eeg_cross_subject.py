"""Subject-held-out EEG pretraining with matched optimization and class-split controls."""
import numpy as np

from eeg_class_alignment import MODEL, PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE, load_subject
from eeg_pooling_dev import evaluate_checkpoint
from eeg_train import DEVICE, OUT, load_json, run_dir, save_json, tensor, train


BASE = OUT / "cross_subject"
SUBJECTS = tuple(range(1, 11))
PROTOCOL = dict(model=MODEL, epochs_pretrain=40, epochs_finetune=100, lr=0.0002, batch=128, temperature=0.07,
                sources="eight subjects excluding both members of each target pair",
                sampling="one source per image per epoch; each block of eight epochs visits all sources once per image",
                preprocess="per subject and channel, statistics computed only from current training rows",
                transfer="encoder weights and batch norm buffers; reset Adam and shuffle at fine-tuning",
                pilot=GATES["pilot"], original_reference="saved seed 0 single-subject 100-epoch result",
                matched_pilot="same pilot gate vs target-only 40+100 epochs, with the same optimizer reset",
                replication=GATES["replication"], replication_reference="target-only 40+100 on the new split",
                replication_split="seed 2027: 200 classes outside the first 200 held-out classes, "
                                  "five support and five query rows per class; all models retrained from scratch",
                replication_folds=[[3,4],[5,6],[7,8],[9,10]], official_test_used=False)


def fresh_split(design):
    labels = design["train_labels"]
    pool = np.setdiff1d(np.unique(labels), design["dev_classes"])
    classes = np.sort(np.random.default_rng(2027).choice(pool, 200, replace=False))
    rows = np.stack([np.flatnonzero(labels == c) for c in classes])
    support, query = rows[:, :5], rows[:, 5:]
    return dict(design, dev_classes=classes, dev_train=np.flatnonzero(~np.isin(labels, classes)),
                dev_query=query.ravel(), dev_support=support.ravel(), dev_labels=np.repeat(np.arange(200), 5),
                dev_centers=design["train_images"][support].mean(1))


def audit(design, name):
    labels, ids = design["train_labels"], design["dev_train"]
    val, support = design["dev_query"], design["dev_support"]
    np.testing.assert_array_equal(np.sort(np.concatenate([ids, val, support])), np.arange(16540))
    assert len(ids) == 14540 and len(val) == len(support) == 1000
    assert len(np.unique(labels[ids])) == 1454 and not np.intersect1d(labels[ids], design["dev_classes"]).size
    np.testing.assert_array_equal(labels[val], design["dev_classes"][design["dev_labels"]])
    np.testing.assert_array_equal(labels[support], np.repeat(design["dev_classes"], 5))
    np.testing.assert_allclose(design["dev_centers"], design["train_images"][support.reshape(200,5)].mean(1))
    np.savez(BASE / f"{name}_split.npz", train=ids, query=val, support=support, classes=design["dev_classes"])


def fit(design, split, targets, sub, seed, matched=False):
    sources = (sub,) if matched else tuple(s for s in SUBJECTS if s not in targets)
    if not matched:
        assert len(sources) == 8 and not set(sources).intersection(targets)
    group = "-".join(str(s) for s in sources)
    prefix = f"cross_subject/{split}"
    pre_stage = f"{prefix}/pretrain/sources-{group}"
    train(MODEL, sources[0], seed, pre_stage, 40, 128, design, sources=sources)
    start = run_dir(pre_stage, MODEL, sources[0], seed) / "checkpoint.pt"
    mode = "target_only" if matched else "transfer"
    stage = f"{prefix}/{mode}"
    train(MODEL, sub, seed, stage, 100, 128, design, split="dev", initial_checkpoint=start)
    folder = run_dir(stage, MODEL, sub, seed)
    path = folder / "validation.npz"
    queries = tensor(load_subject(sub, "training")[design["dev_query"]])
    evaluate_checkpoint(MODEL, sub, seed, design, queries, tensor(design["dev_centers"]), path, stage)
    with np.load(path) as file:
        return file["scores"]


def compare(design, split, matched, targets_list, seeds):
    contrast = "matched" if matched else "historical"
    stage = "pilot" if split == "original" else "replication"
    directory = BASE / split / contrast
    runs = []
    for targets in targets_list:
        for seed in seeds:
            for sub in targets:
                new = fit(design, split, targets, sub, seed)
                if matched:
                    old = fit(design, split, targets, sub, seed, matched=True)
                else:
                    path = OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{sub:02d}.npz"
                    with np.load(path) as source:
                        np.testing.assert_array_equal(source["query_rows"], design["dev_query"])
                        np.testing.assert_array_equal(source["labels"], design["dev_labels"])
                        old = source["scores"]
                folder = directory / stage / f"sub-{sub:02d}" / f"seed-{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                runs.append(paired(sub, seed, folder, design["dev_query"], design["dev_labels"], new, old))
    return dict(summarize(stage, runs, directory), split=split, reference=contrast)


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(exist_ok=True)
    path = BASE / "protocol.json"
    if path.exists() and load_json(path) != PROTOCOL:
        raise ValueError("Existing protocol differs")
    save_json(path, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        design = {k:file[k] for k in ("train_labels", "train_images", "dev_classes", "dev_train", "dev_query",
                                      "dev_support", "dev_centers", "dev_labels")}
    audit(design, "original")
    stages = []
    for split, matched in (("original", False), ("original", True), ("fresh", True)):
        folds, seeds = [(1,2)], [0]
        if split == "fresh":
            design = fresh_split(design)
            audit(design, split)
            folds, seeds = PROTOCOL["replication_folds"], [1,2]
        stages.append(compare(design, split, matched, folds, seeds))
        save_json(BASE / "summary.json", dict(stages=stages, official_test_used=False))
        if not stages[-1]["passed"]:
            break


if __name__ == "__main__":
    main()
