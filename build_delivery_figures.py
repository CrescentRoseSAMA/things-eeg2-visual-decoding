"""Generate five numbered figures and three tables from the formal EEG runs."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

from eeg_evaluate import CONFIGS, SEEDS, SUBJECTS, ranks_and_accuracy, write_csv
from eeg_report import NAMES, figures, read_csv


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEST = OUTPUTS / "delivery_evidence"
MODELS = (*CONFIGS, "ab_pool4")
LABELS = {**NAMES, "ab_pool4": "A+B（4 点池化）"}


def result_tables():
    initial = read_csv(OUTPUTS / "evaluation/runs.csv")
    followup = read_csv(OUTPUTS / "pooling_formal/paired_runs.csv")
    expected = {(model, subject, seed) for model in MODELS for subject in SUBJECTS for seed in SEEDS}
    runs = {}
    for row in initial + followup:
        model = row.get("config", "ab_pool4")
        key = model, int(row["subject"]), int(row["seed"])
        if key in runs:
            raise ValueError(f"Duplicate formal run {key}")
        root = OUTPUTS / ("pooling_formal" if model == "ab_pool4" else "runs")
        folder = root / model / f"sub-{key[1]:02d}" / f"seed-{key[2]}"
        scores = np.load(folder / "scores.npy")
        if scores.shape != (200, 200):
            raise ValueError(f"Incomplete score matrix in {folder}")
        _, top1, top5 = ranks_and_accuracy(scores, np.arange(200))
        real = np.array([100 * top1, 100 * top5])
        recorded = ([float(row["top1_pct"]), float(row["top5_pct"])] if model == "ab_pool4"
                    else [100 * float(row["top1"]), 100 * float(row["top5"])])
        np.testing.assert_allclose(real, recorded, atol=1e-5, rtol=0)
        runs[key] = real
    if set(runs) != expected:
        raise ValueError("Expected 5 configurations × 10 subjects × 5 seeds")

    summary, by_subject = [], []
    per_model = {}
    for model in MODELS:
        cube = np.array([[runs[model, subject, seed] for seed in SEEDS] for subject in SUBJECTS])
        subject_means = cube.mean(axis=1)
        per_model[model] = subject_means
        for subject, (top1, top5) in zip(SUBJECTS, subject_means):
            by_subject.append(dict(配置=LABELS[model], 被试=f"Sub{subject:02d}",
                                   Top1百分比=round(top1, 3), Top5百分比=round(top5, 3)))
        seeds = cube.mean(axis=0)
        mean1, mean5 = seeds.mean(axis=0)
        sd1, sd5 = seeds.std(axis=0, ddof=1)
        group = followup if model == "ab_pool4" else [r for r in initial if r["config"] == model]
        params = {int(r["parameters"]) for r in group}
        if len(params) != 1:
            raise ValueError(f"Inconsistent parameter counts for {model}")
        summary.append(dict(配置=LABELS[model], 次数=len(group), Top1百分比=round(mean1, 5),
                            Top1种子标准差百分点=round(sd1, 5), Top1减15点6百分点=round(mean1 - 15.6, 5),
                            Top5百分比=round(mean5, 5), Top5种子标准差百分点=round(sd5, 5),
                            Top5减42点8百分点=round(mean5 - 42.8, 5), 参数量=params.pop(),
                            训练总小时=round(sum(float(r["train_seconds"]) for r in group) / 3600, 5),
                            显存峰值MiB=max(float(r["peak_gpu_mib"]) for r in group)))
    paired = []
    for full, other in (("ab", "b"), ("ab", "a"), ("ab_pool4", "ab")):
        effects = np.round(per_model[full] - per_model[other], 5)
        paired.append(dict(对照=f"{LABELS[full]} 减 {LABELS[other]}", 配对次数=50,
                           Top1平均差百分点=round(effects[:, 0].mean(), 5),
                           Top1获益人数=int(sum(effects[:, 0] > 0)),
                           Top5平均差百分点=round(effects[:, 1].mean(), 5),
                           Top5获益人数=int(sum(effects[:, 1] > 0))))
    write_csv(DEST / "表1_五种配置正式结果.csv", summary)
    write_csv(DEST / "表2_逐被试五种子均值.csv", by_subject)
    write_csv(DEST / "表3_配对消融与池化对照.csv", paired)


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEST.mkdir(exist_ok=True, parents=True)
    figures(SimpleNamespace(data=ROOT / "data/THINGS-EEG2", dev=OUTPUTS / "dev", runs=OUTPUTS / "runs"), DEST)
    # The two heatmaps use the same subject, seed, colors and score scale.
    original = np.load(DEST / "图4_EEG与候选模板相似度.npy")
    followup = np.load(OUTPUTS / "pooling_formal/ab_pool4/sub-01/seed-0/scores.npy")
    low, high = min(original.min(), followup.min()), max(original.max(), followup.max())
    for stem, title, scores in (("图4_EEG与候选模板相似度", "A+B, 8-sample pooling", original),
                                ("图5_四点池化EEG与候选模板相似度", "A+B, 4-sample pooling", followup)):
        np.save(DEST / (stem + ".npy"), scores)
        fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
        mesh = ax.imshow(scores, cmap="viridis", interpolation="nearest", vmin=low, vmax=high)
        ax.set(xlabel="Template index", ylabel="EEG query index", title=f"Sub01, seed 0: {title}")
        fig.colorbar(mesh, ax=ax, label="Matching score")
        fig.savefig(DEST / (stem + ".png"), dpi=170)
        plt.close(fig)
    result_tables()
    print(f"Verified 250 formal scores and exported 5 figures and 3 tables in {DEST}")


if __name__ == "__main__":
    main()
