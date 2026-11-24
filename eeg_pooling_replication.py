"""Replicate the temporal-pooling experiment on eight additional subjects."""
import numpy as np
import torch

from eeg_data import CACHE, load_subject
from eeg_pooling_dev import evaluate_checkpoint
from eeg_train import OUT, load_json, run_dir, save_json, tensor, train, write_log


SUBJECTS = range(3, 11)
SEEDS = (1, 2)
MODELS = ("ab", "ab_pool4", "ab_pool8_stride4")


def summarize(rows, output):
    pairs = {(r["subject"], r["seed"], r["config"]): r for r in rows}
    assert len(pairs) == len(rows) == 48
    differences = []
    summary = {"means_pct": {}, "contrasts": {}}
    for model in MODELS:
        group = [r for r in rows if r["config"] == model]
        summary["means_pct"][model] = {
            k: round(float(np.mean([r[k] for r in group])), 5) for k in ("top1_pct", "top5_pct")}
    for reference in ("ab", "ab_pool8_stride4"):
        delta = np.empty((len(SUBJECTS), len(SEEDS), 2))
        for i, subject in enumerate(SUBJECTS):
            for j, seed in enumerate(SEEDS):
                short = pairs[subject, seed, "ab_pool4"]
                other = pairs[subject, seed, reference]
                delta[i, j] = [round(short[k] - other[k], 5) for k in ("top1_pct", "top5_pct")]
                differences.append(dict(reference=reference, subject=subject, seed=seed,
                                        top1_delta_pp=delta[i,j,0], top5_delta_pp=delta[i,j,1]))
        seed_mean = delta.mean(axis=0)
        subject_mean = delta.mean(axis=1).round(5)
        mean = delta.mean(axis=(0, 1)).round(5)
        wins = (subject_mean > 0).sum(axis=0)
        summary["contrasts"][reference] = dict(mean_delta_pp=mean.tolist(), wins=wins.tolist(),
            seed_mean_delta_pp=seed_mean.round(5).tolist(), subject_mean_delta_pp=subject_mean.tolist(),
            passes_rule=bool((seed_mean[:,0].round(5) > 0).all() and wins[0] >= 5 and mean[1] >= 0))
    write_log(output / "paired_differences.csv", differences)
    save_json(output / "summary.json", summary)


def main():
    output = OUT / "pooling_replication"
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(scope="development only, previous 200 held-out training concepts",
                    subjects=list(SUBJECTS), seeds=list(SEEDS), configs=list(MODELS), epochs=100,
                    batch_size=128, learning_rate=0.0002, temperature=0.07, readout="epoch 100",
                    primary="ab_pool4 minus ab", window_control="ab_pool4 minus ab_pool8_stride4",
                    overlap="8-point window, stride 4, symmetric padding 2; edge means exclude padding",
                    rule="Each seed's across-subject Top-1 delta >0, >=5/8 positive subject mean Top-1 "
                         "deltas, pooled mean Top-5 delta >=0. Apply separately to each contrast.",
                    metrics=["Top-1", "Top-5"], design="paired within subject and seed")
    manifest = output / "protocol.json"
    if manifest.exists() and load_json(manifest) != protocol:
        raise ValueError("Existing protocol differs")
    save_json(manifest, protocol)
    with np.load(CACHE / "design.npz") as file:
        design = {k: file[k] for k in ("train_images", "dev_train", "dev_query", "dev_centers", "dev_labels")}
        assert not np.intersect1d(file["train_labels"][design["dev_train"]], file["train_labels"][design["dev_query"]]).size
        assert len(design["dev_query"]) == 1000 and len(design["dev_centers"]) == 200
    centers = tensor(design["dev_centers"])
    rows = []
    for subject in SUBJECTS:
        queries = tensor(load_subject(subject, "training")[design["dev_query"]])
        for seed in SEEDS:
            mean_ref = std_ref = None
            for config in MODELS:
                train(config, subject, seed, "dev", 100, 128, design)
                destination = output / f"{config}_sub-{subject:02d}_seed-{seed}.npz"
                row, mu, std = evaluate_checkpoint(config, subject, seed, design, queries, centers, destination)
                if mean_ref is not None:
                    torch.testing.assert_close(mu, mean_ref)
                    torch.testing.assert_close(std, std_ref)
                mean_ref, std_ref = mu, std
                row["train_seconds"] = load_json(run_dir("dev", config, subject, seed) / "training.json")["train_seconds"]
                rows.append(row)
                write_log(output / "runs.csv", rows)
    summarize(rows, output)
    print("All 48 development runs reloaded and scored.", flush=True)


if __name__ == "__main__":
    main()
