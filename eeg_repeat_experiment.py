"""Development-only trial subsampling experiment with the original matched pooling controls."""
import csv

import numpy as np

from eeg_class_alignment import PROTOCOL as PRIOR_PROTOCOL, development
from eeg_data import CACHE, CHANNELS, DATA, load_subject
from eeg_repeat_views import omitted_means
from eeg_train import DEVICE, OUT, load_json, save_json


BASE = OUT / "repeat_sampling"
PROTOCOL = dict(model="ab_pool4", loss="original symmetric image InfoNCE", epochs=100, batch=128,
                adam_lr=0.0002, temperature=0.07, image_visits_per_epoch=1,
                training="p=1/2 four-repeat mean; otherwise uniformly omit one of four real repeats",
                sampling="independent NumPy default_rng([seed, one_based_epoch, 304]); dev_train row order",
                normalization="same training four-repeat mean and std per channel as control",
                evaluation="unchanged four-repeat EEG, 1000 development queries, fixed epoch 100",
                pilot=PRIOR_PROTOCOL["pilot"], replication=PRIOR_PROTOCOL["replication"],
                formal_test_used=False)


def raw_trials(sub):
    path = DATA / "Preprocessed_data" / f"sub-{sub:02d}" / "preprocessed_eeg_training.npy"
    packet = np.load(path, allow_pickle=True).item()
    raw = packet["preprocessed_eeg_data"]
    if raw.shape != (16540, 4, 17, 100) or list(packet["ch_names"]) != CHANNELS:
        raise ValueError(f"Unexpected EEG layout for subject {sub}")
    np.testing.assert_allclose(packet["times"], np.arange(-20, 80) / 100, atol=1e-8)
    return raw[:, :, :, 20:]


def audit(sub):
    raw = raw_trials(sub)
    full = load_subject(sub)
    for i in range(0, len(raw), 128):
        actual = raw[i:i+128].mean(1).astype(np.float32)
        if not np.isfinite(actual).all():
            raise ValueError(f"Non-finite raw mean for subject {sub} at {i}")
        np.testing.assert_array_equal(actual, full[i:i+128])
    save_json(BASE / f"audit_sub-{sub:02d}.json", dict(subject=sub, exact_four_repeat_rows=len(raw),
              channels=17, time_points=80, repeats=4))


def load_views(sub, rows):
    raw = raw_trials(sub)
    out = np.empty((len(rows), 4, 17, 80), dtype=np.float32)
    for i in range(0, len(rows), 128):
        out[i:i+128] = omitted_means(raw[rows[i:i+128]])
    return out


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing repeat sampling protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        design = {k: file[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels",
                                       "train_images", "train_labels")}
    with (CACHE / "sample_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    if len(rows) != 16540:
        raise ValueError("Training index length differs")
    for i, row in enumerate(rows):
        if (int(row["eeg_row"]), int(row["image_feature_row"]), int(row["split_class_0based"])) != (
                i, i, int(design["train_labels"][i])):
            raise ValueError(f"Image and EEG index mismatch at {i}")
    np.testing.assert_array_equal(design["dev_train"], [i for i,r in enumerate(rows) if r["role"] == "dev_train"])
    stages = []
    for stage, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        for sub in subjects:
            audit(sub)
        stages.append(development(stage, subjects, seeds, design, BASE, load_views))
        if not stages[-1]["passed"]:
            break
    save_json(BASE / "summary.json", dict(stopped_at=stage, formal_test_used=False, stages=stages))


if __name__ == "__main__":
    main()
