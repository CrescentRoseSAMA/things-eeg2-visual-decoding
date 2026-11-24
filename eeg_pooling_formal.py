"""Locked formal evaluation of the 4-point pooling follow-up on THINGS-EEG2."""
import numpy as np

from eeg_data import CACHE
from eeg_evaluate import ranks_and_accuracy
from eeg_train import OUT, load_json, read_log, reload_and_score, run_dir, save_json, train, write_log


STAGE = "pooling_formal"
MODEL = "ab_pool4"
SUBJECTS = range(1, 11)
SEEDS = range(5)


def checked_metrics(folder):
    scores = np.load(folder / "scores.npy")
    assert scores.shape == (200, 200)
    _, top1, top5 = ranks_and_accuracy(scores, np.arange(200))
    metrics = load_json(folder / "metrics.json")
    np.testing.assert_allclose([top1, top5], [metrics["top1"], metrics["top5"]], atol=1e-6, rtol=0)
    assert int(metrics["batch_size"]) == 128 and int(metrics["trained_epochs"]) == 100
    assert [int(row["epoch"]) for row in read_log(folder / "train.csv")] == list(range(1, 101))
    return metrics, np.round(np.array([top1, top5]) * 100, 6)


def main():
    output = OUT / STAGE
    output.mkdir(exist_ok=True)
    protocol = dict(model=MODEL, reference="original ab checkpoints and paired scores", subjects=list(SUBJECTS),
                    seeds=list(SEEDS), epochs=100, batch_size=128, learning_rate=0.0002, temperature=0.07,
                    fit="all 1654 official training classes, within subject", evaluate="fixed 200 published templates",
                    readout="epoch 100; all 50 fits finish before test evaluation",
                    selection="development replication primary contrast passed; equal-capacity contrast did not",
                    endpoints=["Top-1", "Top-5"], target_pct=[15.6, 42.8],
                    aggregation="average ten subjects per seed, then mean and sample SD across seeds",
                    paired_effect="ab_pool4 minus original ab in percentage points, identical subject and seed",
                    interpretation="follow-up evaluation reuses the previously reported official test classes")
    manifest = output / "protocol.json"
    if manifest.exists() and load_json(manifest) != protocol:
        raise ValueError("Existing follow-up protocol differs")
    save_json(manifest, protocol)
    with np.load(CACHE / "design.npz") as design_file:
        training = {"train_images": design_file["train_images"]}
    for subject in SUBJECTS:
        for seed in SEEDS:
            train(MODEL, subject, seed, STAGE, 100, 128, training)
    del training
    with np.load(CACHE / "design.npz") as design_file:
        testing = {k: design_file[k] for k in ("test_centers", "test_labels")}
    results = np.empty((10, 5, 2))
    reference = np.empty_like(results)
    rows = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            reload_and_score(MODEL, subject, seed, testing, stage=STAGE)
            m, new = checked_metrics(run_dir(STAGE, MODEL, subject, seed))
            prior, old = checked_metrics(run_dir("runs", "ab", subject, seed))
            assert (m["config"], prior["config"]) == (MODEL, "ab")
            assert m["subject"] == prior["subject"] == subject and m["seed"] == prior["seed"] == seed
            results[subject-1, seed], reference[subject-1, seed] = new, old
            rows.append(dict(subject=subject, seed=seed, top1_pct=new[0], top5_pct=new[1],
                             old_top1_pct=old[0], old_top5_pct=old[1], top1_delta_pp=new[0]-old[0],
                             top5_delta_pp=new[1]-old[1], parameters=m["parameters"],
                             train_seconds=m["train_seconds"], peak_gpu_mib=m["peak_gpu_mib"]))
    write_log(output / "paired_runs.csv", rows)
    by_seed = results.mean(axis=0)
    delta = results-reference
    subject_delta = delta.mean(axis=1)
    summary = dict(runs=len(rows), metrics=["Top-1", "Top-5"], mean_pct=by_seed.mean(0).round(5).tolist(),
                   seed_sd_pp=by_seed.std(0, ddof=1).round(5).tolist(),
                   reference_pct=reference.mean(axis=(0,1)).round(5).tolist(),
                   delta_vs_ab_pp=delta.mean(axis=(0,1)).round(5).tolist(),
                   delta_vs_course_pp=(by_seed.mean(0)-[15.6,42.8]).round(5).tolist(),
                   subject_delta_pp=subject_delta.round(5).tolist(),
                   subjects_improved=(subject_delta.round(5)>0).sum(0).tolist(),
                   seed_mean_pct=by_seed.round(5).tolist())
    save_json(output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
