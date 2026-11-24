"""Matched control experiment for visually similar, different-class training negatives."""
import numpy as np

from eeg_class_alignment import MODEL, PROTOCOL as GATES, paired, summarize
from eeg_cross_subject import fresh_split
from eeg_data import CACHE, load_subject
from eeg_pooling_dev import evaluate_checkpoint
from eeg_train import DEVICE, OUT, load_json, run_dir, save_json, tensor, train, write_log
from eeg_visual_negatives import ImageNegatives, neighbors


BASE = OUT / "visual_negatives"
PROTOCOL = dict(model=MODEL, batch=128, epochs=100, adam_lr=0.0002, temperature=0.07,
                extra="16 distinct different classes, one outside-batch image per class, EEG-to-image only",
                near="sample from 20 nearest eligible training class centroids by cosine",
                random="uniform over all eligible different training classes; same number of negatives",
                schedule="independent epoch RNG, class draws shared by class per epoch, image drawn per anchor",
                endpoint="epoch 100, original scoring", pilot=GATES["pilot"], replication=GATES["replication"],
                gate="near vs baseline passes stage's usual gate AND aggregate near Top1 hits > random "
                     "AND near has fewer wrong Top1 predictions into the true class's 20 nearest templates than random",
                replication_split="fresh development classes, split seed 2027, all three arms fit from scratch",
                formal_test_used=False)


def fit(d, stage, arm, subject, seed, sampler, x):
    name = f"{BASE.name}/{stage}/{arm}"
    train(MODEL, subject, seed, name, 100, 128, d, split="dev", negatives=sampler)
    folder = run_dir(name, MODEL, subject, seed)
    path = folder / "validation.npz"
    evaluate_checkpoint(MODEL, subject, seed, d, x, tensor(d["dev_centers"]), path, name)
    with np.load(path) as saved:
        return saved["scores"]


def run(stage, d, subjects, seeds):
    idx, rows, labels = d["dev_train"], d["dev_query"], d["dev_labels"]
    assert len(np.unique(idx)) == len(idx) == 14540 and len(rows) == len(labels) == 1000
    assert not np.intersect1d(d["train_labels"][idx], d["dev_classes"]).size
    np.testing.assert_array_equal(d["train_labels"][rows], d["dev_classes"][labels])
    pools = {arm: ImageNegatives(d["train_images"][idx], d["train_labels"][idx], arm)
             for arm in ("random", "near")}
    near_classes = neighbors(d["dev_centers"].astype(np.float64))[:, :20]
    details, vs_base, vs_random = [], [], []
    for sub in subjects:
        x = tensor(load_subject(sub, "training")[rows])
        for seed in seeds:
            if stage == "pilot":
                path = OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{sub:02d}.npz"
                with np.load(path) as ref:
                    np.testing.assert_array_equal(ref["query_rows"], rows)
                    np.testing.assert_array_equal(ref["labels"], labels)
                    baseline = ref["scores"]
            else:
                baseline = fit(d, stage, "baseline", sub, seed, None, x)
            scores = dict(baseline=baseline)
            for arm, sampler in pools.items():
                scores[arm] = fit(d, stage, arm, sub, seed, sampler, x)
            for arm, arr in scores.items():
                pred = arr.argmax(1)
                wrong = int((pred != labels).sum())
                near_errors = int((near_classes[labels] == pred[:, None]).any(1).sum())
                top5 = np.argsort(-arr, axis=1, kind="stable")[:, :5]
                details.append(dict(subject=sub, seed=seed, arm=arm, top1_hits=len(labels)-wrong,
                                    top5_hits=int((top5==labels[:,None]).any(1).sum()), errors=wrong,
                                    neighbor_errors=near_errors, other_errors=wrong-near_errors))
            for reference, runs in (("baseline", vs_base), ("random", vs_random)):
                folder = BASE / stage / f"vs_{reference}" / f"sub-{sub:02d}" / f"seed-{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                runs.append(paired(sub, seed, folder, rows, labels, scores["near"], scores[reference]))
            write_log(BASE / stage / "arm_counts.csv", details)
    normal = summarize(stage, vs_base, BASE / "vs_baseline")
    write_log(BASE / stage / "paired_vs_random.csv", vs_random)
    near_errors = {arm: sum(r["neighbor_errors"] for r in details if r["arm"] == arm)
                   for arm in ("baseline", "random", "near")}
    return dict(stage=stage, baseline_comparison=normal,
                top1_hits_delta_vs_random=sum(r["gain1"] for r in vs_random), neighbor_errors=near_errors,
                passed=normal["passed"] and sum(r["gain1"] for r in vs_random)>0
                       and near_errors["near"]<near_errors["random"])


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing sampling protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        d = {k:file[k] for k in ("dev_train","dev_query","dev_classes","dev_centers","dev_labels",
                                 "train_images","train_labels")}
    stages = []
    for stage, subjects, seeds in (("pilot", (1,2), (0,)), ("replication", range(3,11), (1,2))):
        if stage == "replication":
            d = fresh_split(d)
        (BASE / "vs_baseline" / stage).mkdir(parents=True, exist_ok=True)
        stages.append(run(stage, d, subjects, seeds))
        save_json(BASE / "summary.json", dict(stages=stages, official_test_used=False))
        if not stages[-1]["passed"]:
            break


if __name__ == "__main__":
    main()
