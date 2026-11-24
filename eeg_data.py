"""Prepare row-checked THINGS-EEG2 17-channel data for EEG/CLIP retrieval.

Run python eeg_data.py from any directory. Raw inputs go under data/THINGS-EEG2.
The consumer fits normalization using only training rows for its split.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data" / "THINGS-EEG2"
CACHE = DATA / "derived"
FIGURES = PROJECT / "reports" / "figures"
CHANNELS = ["Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1",
            "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"]
METADATA_SHA = "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c"
FEATURE_SHA = {
    "center_clip.npy": "40286760c0559dcc644aa13e9914508b2ec55a795cee46473e786e61a4406060",
    "clip_feature_maps_test.npy": "7bc121ec7ee017cb0f06b927204e28ebce70d852e4c690730230b92c7944a23d",
    "clip_feature_maps_training.npy": "85fe0038b117d99e2fcba3f83e8eb7fbd81bc32b3c07a7056dbb6789f57b944e",
}


def verify_hash(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch against published artifact: {path}")


def condition_rows(concepts, files, n_classes, per_class):
    """Respect acquisition order; reject permutations instead of silently sorting."""
    pairs = list(zip(concepts, files))
    if (len(concepts) != len(files) or len(pairs) != n_classes * per_class
            or pairs != sorted(pairs) or len(set(pairs)) != len(pairs)):
        raise ValueError("Image count, uniqueness, or acquisition row order mismatch")
    names, labels, counts = np.unique(concepts, return_inverse=True, return_counts=True)
    if len(names) != n_classes or not np.all(counts == per_class):
        raise ValueError("Unexpected number of concepts or images per concept")
    return labels.astype(np.int64)


def development_split(labels, train_images):
    heldout = np.sort(np.random.default_rng(2026).choice(1654, 200, replace=False))
    # Rows within each concept are already checked against ascending stimulus ID.
    rows = np.stack([np.flatnonzero(labels == c) for c in heldout])
    support, queries = rows[:, :5], rows[:, 5:]
    return {
        "dev_classes": heldout, "dev_support": support.ravel(),
        "dev_train": np.flatnonzero(~np.isin(labels, heldout)),
        "dev_query": queries.ravel(), "dev_labels": np.repeat(np.arange(200), 5),
        "dev_centers": train_images[support].mean(axis=1),
    }


def load_design():
    """Return image vectors, raw template means, indices and labels (0-based)."""
    with np.load(CACHE / "design.npz", allow_pickle=False) as data:
        return dict(data)


def load_subject(subject, split="training"):
    """Memory map float32 [image, electrode, post-stimulus time], without scaling."""
    if split not in ("training", "test") or not 1 <= subject <= 10:
        raise ValueError("Expected subject 1..10 and split training or test")
    return np.load(CACHE / f"sub-{subject:02d}_{split}.npy", mmap_mode="r")


def prepare_design():
    metadata_path = DATA / "image_metadata.npy"
    verify_hash(metadata_path, METADATA_SHA)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    image_dir = DATA / "dnn_feature"
    for name, sha in FEATURE_SHA.items():
        verify_hash(image_dir / name, sha)
    concepts = {p: metadata[p + "_img_concepts_THINGS"] for p in ("train", "test")}
    if set(concepts["train"]) & set(concepts["test"]):
        raise ValueError("Official global THINGS concept present in both splits")
    row_labels = {}
    for p, count, per_class in (("train", 1654, 10), ("test", 200, 1)):
        row_labels[p] = condition_rows(metadata[p + "_img_concepts"], metadata[p + "_img_files"],
                                       count, per_class)
    images = np.load(image_dir / "clip_feature_maps_training.npy")[:, 0, :]
    centers = np.load(image_dir / "center_clip.npy")
    if images.shape != (16540, 768) or centers.shape != (200, 768):
        raise ValueError("Unexpected CLIP array shapes")
    if not np.isfinite(images).all() or not np.isfinite(centers).all():
        raise ValueError("Nonfinite image vectors or class templates")
    images = images / np.linalg.norm(images, axis=1, keepdims=True)
    design = development_split(row_labels["train"], images)
    design.update(train_images=images, test_centers=centers,
                  train_labels=row_labels["train"], test_labels=row_labels["test"])
    with (CACHE / "design.partial").open("wb") as stream:
        np.savez(stream, **design)
    (CACHE / "design.partial").replace(CACHE / "design.npz")
    roles = np.full(16540, "dev_train", dtype="U12")
    roles[design["dev_support"]] = "dev_support"
    roles[design["dev_query"]] = "dev_query"
    with (CACHE / "sample_index.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("split", "stimulus_id_1based", "eeg_row", "image_feature_row",
                         "split_class_0based", "test_template_row", "concept_dir",
                         "things_concept", "image_filename", "role"))
        for p in ("train", "test"):
            for row, (name, filename) in enumerate(zip(metadata[p + "_img_concepts"],
                                                      metadata[p + "_img_files"])):
                label = int(row_labels[p][row])
                writer.writerow((p, row + 1, row, row, label, label if p == "test" else "",
                                 name, concepts[p][row], filename, roles[row] if p == "train" else "test"))
    print("Verified 16,540 training rows, 200 test rows and published image hashes.", flush=True)


def waveform(times, waves):
    """First training image (aardvark_01b.jpg), Sub01 Oz, before trial averaging."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    mean = waves.mean(axis=0)
    np.savetxt(FIGURES / "fig01_eeg_waveform.csv", np.column_stack((times, waves.T, mean)),
               delimiter=",", header="time_s,repeat_1,repeat_2,repeat_3,repeat_4,mean", comments="")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    for i, wave in enumerate(waves):
        ax.plot(times, wave, color="#90a4ae", lw=0.75, alpha=0.8,
                label="Individual repeats" if i == 0 else None)
    ax.plot(times, mean, color="#164e78", lw=1.9, label="Four-repeat mean")
    ax.axvline(0, color="#333333", ls="--", lw=0.9, label="Stimulus onset")
    ax.set(xlabel="Time from image onset (s)", ylabel="Preprocessed EEG (a.u.)",
           title="Sub01 · Oz · aardvark_01b.jpg")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(FIGURES / "fig01_eeg_waveform.png", dpi=220)
    plt.close(fig)


def cache_subject(subject, split):
    raw_path = DATA / "Preprocessed_data" / f"sub-{subject:02d}" / f"preprocessed_eeg_{split}.npy"
    packet = np.load(raw_path, allow_pickle=True).item()
    times = np.asarray(packet["times"])
    values = packet["preprocessed_eeg_data"]
    n_images, repeats = (16540, 4) if split == "training" else (200, 80)
    if values.shape != (n_images, repeats, 17, 100) or list(packet["ch_names"]) != CHANNELS:
        raise ValueError(f"EEG tensor shape or electrode list differs: {raw_path}")
    np.testing.assert_allclose(times, np.arange(-20, 80) / 100, atol=1e-8)
    keep = np.flatnonzero((times >= 0) & (times < 0.8))
    np.testing.assert_array_equal(keep, np.arange(20, 100))
    filename = CACHE / f"sub-{subject:02d}_{split}.npy"
    pending = filename.with_suffix(".partial")
    out = np.lib.format.open_memmap(str(pending), "w+", dtype=np.float32, shape=(n_images, 17, 80))
    for start in range(0, n_images, 128):
        # Slice before reducing to avoid allocating a copy of the whole EEG packet.
        group = values[start:start + 128, :, :, 20:]
        mean = group.mean(axis=1)
        if not np.isfinite(mean).all():
            raise ValueError(f"Nonfinite EEG average in {raw_path} at row {start}")
        out[start:start + len(mean)] = mean
    del out
    pending.replace(filename)
    if subject == 1 and split == "training":
        waveform(times, values[0, :, CHANNELS.index("Oz")])
    print(f"Prepared {filename.name}: {n_images} x 17 x 80", flush=True)


def audit_inputs(subjects):
    """Reload independent raw reference rows and verify the generated artifacts."""
    design = load_design()
    np.testing.assert_array_equal(design["test_centers"], np.load(DATA / "dnn_feature/center_clip.npy"))
    checks = []
    for subject in subjects:
        for split in ("training", "test"):
            path = DATA / "Preprocessed_data" / f"sub-{subject:02d}" / f"preprocessed_eeg_{split}.npy"
            source = np.load(path, allow_pickle=True).item()
            cached = load_subject(subject, split)
            rows = [0, len(cached) // 2, len(cached) - 1]
            expected = source["preprocessed_eeg_data"][rows].mean(axis=1)[:, :, 20:].astype(np.float32)
            np.testing.assert_array_equal(cached[rows], expected)
            checks.append({"subject": subject, "split": split, "rows_verified": rows})
            del source, cached
    with (CACHE / "sample_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    identifiers = {(r["split"], r["stimulus_id_1based"]) for r in index}
    if len(identifiers) != len(index) or len(index) != 16740:
        raise ValueError("Expected 16,740 distinct indexed stimulus conditions")
    audit = {
        "unique_indexed_conditions": len(identifiers), "channels": CHANNELS,
        "sampling_hz": 100, "first_last_model_times_s": [0.0, 0.79],
        "metadata_sha256": METADATA_SHA, "image_feature_sha256": FEATURE_SHA,
        "source_cache_row_equality": checks, "published_test_centers_equal": True,
        "upstream_whitening": "The published Gifford session-level MVNN averages training and test covariance.",
        "upstream_preprocessing": "https://github.com/gifale95/eeg_encoding/blob/main/02_eeg_preprocessing/preprocessing_utils.py#L125-L166",
        "condition_order_source": "https://github.com/gifale95/eeg_encoding/issues/3",
    }
    destination = PROJECT / "reports" / "data_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks) * 3} source-to-cache rows and {len(index)} indexed conditions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--audit-only", action="store_true", help="Independently verify prepared files")
    args = parser.parse_args()
    if args.audit_only:
        audit_inputs(args.subjects)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    prepare_design()
    for subject in args.subjects:
        for split in ("training", "test"):
            cache_subject(subject, split)
    print("Completed subjects: " + ", ".join(map(str, args.subjects)))


if __name__ == "__main__":
    main()
