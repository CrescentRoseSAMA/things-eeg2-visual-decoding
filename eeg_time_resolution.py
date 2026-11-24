"""Paired, training-partition-only comparison of two EEG time resolutions."""
import csv
import json
import os
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np

from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy


OUT = Path(__file__).resolve().parent / "outputs" / "development_time_resolution"
ALPHA = 0.1


def features(eeg, width):
    n, channels, times = eeg.shape
    return eeg.reshape(n, channels, times // width, width).mean(axis=3, dtype=np.float64).reshape(n, -1)


def fit(eeg, images, width):
    x = features(eeg, width)
    center = x.mean(axis=0)
    scale = x.std(axis=0).clip(1e-8)
    x -= center
    x /= scale
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += ALPHA * len(x)
    targets = np.asarray(images, dtype=np.float64)
    rhs = x.T @ (targets - targets.mean(axis=0))
    # Fold the training-only scaling into the raw-coordinate coefficients.
    return center, np.linalg.solve(gram, rhs) / scale[:, None]


def predict(eeg, width, center, weights, templates):
    y = (features(eeg, width) - center) @ weights
    y /= np.linalg.norm(y, axis=1, keepdims=True).clip(1e-12)
    return y @ templates.T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(split="development", subjects=list(range(1, 11)), widths=[8, 1],
                    input_hz=100, alpha=ALPHA, fixed_split_seed=2026,
                    score="L2-normalized predicted centered CLIP feature dot original development template",
                    target="training image vector minus training image-vector mean",
                    comparison="full80 minus pooled10 in percentage points, paired by subject",
                    primary="10-subject mean Top-1 difference", secondary="mean Top-5 difference")
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with np.load(CACHE / "design.npz") as design:
        train, query = design["dev_train"], design["dev_query"]
        labels, templates = design["dev_labels"], design["dev_centers"]
        images = design["train_images"][train]
        assert (len(train), len(query), len(templates)) == (14540, 1000, 200)
        assert not np.intersect1d(design["train_labels"][train], design["train_labels"][query]).size
    rows = []
    for subject in range(1, 11):
        eeg = load_subject(subject, "training")
        x, q = eeg[train], eeg[query]
        for config, width in (("pooled10", 8), ("full80", 1)):
            folder = OUT / config / f"sub-{subject:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            start = perf_counter()
            center, weights = fit(x, images, width)
            seconds = perf_counter() - start
            scores = predict(q, width, center, weights, templates)
            ranks, top1, top5 = ranks_and_accuracy(scores, labels)
            np.savez(folder / "weights.npz", center=center, weights=weights, width=width)
            np.save(folder / "scores.npy", scores)
            with np.load(folder / "weights.npz") as saved:
                replay = predict(q, int(saved["width"]), saved["center"], saved["weights"], templates)
            np.testing.assert_array_equal(replay, scores)
            np.save(folder / "ranks.npy", ranks)
            rows.append(dict(subject=subject, config=config, inputs=len(center), fit_seconds=seconds,
                             top1_pct=100*top1, top5_pct=100*top5))
            print(rows[-1], flush=True)
    for name, records in (("runs", rows), ("paired_differences", [
        dict(subject=a["subject"], top1_pp=b["top1_pct"]-a["top1_pct"],
             top5_pp=b["top5_pct"]-a["top5_pct"]) for a, b in zip(rows[::2], rows[1::2])
    ])):
        with (OUT / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
