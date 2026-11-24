"""Recompute formal THINGS-EEG2 rankings and summarize complete paired runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


CONFIGS = ("baseline", "a", "b", "ab")
SUBJECTS = range(1, 11)
SEEDS = range(5)
FIELDS = ("config", "subject", "seed", "top1", "top5", "parameters",
          "train_seconds", "peak_gpu_mib", "epochs", "batch_size")


def ranks_and_accuracy(scores, labels):
    """Return one-based ranks and Top-k proportions. Column order breaks ties."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 5
            or not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all()):
        raise ValueError("Expected a finite query-by-template matrix with at least five columns")
    if (labels.shape != (len(scores),) or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any() or (labels >= scores.shape[1]).any()):
        raise ValueError("Expected one valid integer template index per query")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = (order == labels[:, None]).argmax(axis=1) + 1
    return ranks, float((ranks == 1).mean()), float((ranks <= 5).mean())


def collect(root):
    rows, pending = [], []
    for config in CONFIGS:
        for subject in SUBJECTS:
            for seed in SEEDS:
                run = root / config / f"sub-{subject:02d}" / f"seed-{seed}"
                files = ("metrics.json", "scores.npy", "train.csv", "checkpoint.pt")
                missing = [name for name in files if not (run / name).is_file()
                           or (run / name).stat().st_size == 0]
                if missing:
                    pending.append(dict(config=config, subject=subject, seed=seed,
                                        missing=", ".join(missing)))
                    continue
                data = json.loads((run / "metrics.json").read_text(encoding="utf-8-sig"))
                if tuple(data[k] for k in ("config", "subject", "seed")) != (config, subject, seed):
                    raise ValueError(f"Run key mismatch in {run}")
                scores = np.load(run / "scores.npy", allow_pickle=False)
                if scores.shape != (200, 200):
                    raise ValueError(f"Expected a 200-by-200 formal score matrix in {run}")
                _, top1, top5 = ranks_and_accuracy(scores, np.arange(200))
                if not np.allclose([data["top1"], data["top5"]], [top1, top5],
                                   rtol=0, atol=1e-6):
                    raise ValueError(f"Recorded metrics disagree with stored scores in {run}")
                if not all(np.isfinite(data[k]) and data[k] > 0 for k in
                           ("epochs", "batch_size", "parameters", "train_seconds")):
                    raise ValueError(f"Invalid training metadata in {run}")
                if not np.isfinite(data["peak_gpu_mib"]) or data["peak_gpu_mib"] < 0:
                    raise ValueError(f"Invalid GPU memory in {run}")
                with (run / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
                    epochs = list(csv.DictReader(stream))
                if not epochs or [int(e["epoch"]) for e in epochs] != list(range(1, data["epochs"] + 1)):
                    raise ValueError(f"Incomplete epoch log in {run}")
                if not all(np.isfinite(float(e["loss"])) for e in epochs):
                    raise ValueError(f"Invalid training loss in {run}")
                data.update(top1=top1, top5=top5)
                rows.append({k: data[k] for k in FIELDS})
    return rows, pending


def aggregate(rows):
    """Aggregate only the balanced four-configuration factorial matrix."""
    if len(rows) != len(CONFIGS) * len(SUBJECTS) * len(SEEDS):
        return [], [], [], []
    by_key = {(r["config"], r["subject"], r["seed"]): r for r in rows}
    overall, subjects, seeds, deltas = [], [], [], []
    for config in CONFIGS:
        group = [r for r in rows if r["config"] == config]
        group_seeds = []
        for seed in SEEDS:
            seed_row = dict(config=config, seed=seed)
            for k in ("top1", "top5"):
                seed_row[k + "_pct"] = 100 * mean(by_key[config, s, seed][k] for s in SUBJECTS)
            group_seeds.append(seed_row)
        seeds.extend(group_seeds)
        for subject in SUBJECTS:
            summary = dict(config=config, subject=subject)
            for k in ("top1", "top5"):
                summary[k + "_pct"] = 100 * mean(by_key[config, subject, n][k] for n in SEEDS)
            subjects.append(summary)
        summary = dict(config=config)
        for k, reference in (("top1", 15.6), ("top5", 42.8)):
            values = [r[k + "_pct"] for r in group_seeds]
            summary[k + "_pct"] = mean(values)
            summary[k + "_seed_sd_pp"] = stdev(values)
            summary[k + "_vs_reference_pp"] = mean(values) - reference
        if len({r["parameters"] for r in group}) != 1:
            raise ValueError(f"Different model sizes within {config}")
        summary.update(parameters=group[0]["parameters"],
                       train_hours=sum(r["train_seconds"] for r in group) / 3600,
                       peak_gpu_mib=max(r["peak_gpu_mib"] for r in group))
        overall.append(summary)
    for other in ("b", "a"):
        for subject in SUBJECTS:
            for seed in SEEDS:
                full, reduced = by_key["ab", subject, seed], by_key[other, subject, seed]
                delta = dict(comparison="ab-" + other, subject=subject, seed=seed)
                for k in ("top1", "top5"):
                    delta[k + "_delta_pp"] = 100 * (full[k] - reduced[k])
                deltas.append(delta)
    return overall, subjects, seeds, deltas


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--report", action="store_true", help="Generate figures and Word after all 200 runs")
    parser.add_argument("--dev", type=Path, default=Path("outputs/dev"))
    parser.add_argument("--data", type=Path, default=Path("data/THINGS-EEG2"))
    parser.add_argument("--name", default="")
    parser.add_argument("--student-id", default="")
    parser.add_argument("--college", default="")
    parser.add_argument("--class-name", default="")
    parser.add_argument("--debug-notes", type=Path, help="UTF-8 notes on actual debugging")
    parser.add_argument("--debug-image", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    rows, pending = collect(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "runs.csv", rows, FIELDS)
    write_csv(args.out / "pending.csv", pending, ("config", "subject", "seed", "missing"))
    status = dict(complete=len(rows), expected=200, pending=len(pending))
    (args.out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"{len(rows)}/200 formal runs verified")
    if pending:
        if args.report:
            parser.exit(2, "Incomplete matrix: submission report was not generated. See pending.csv\n")
        return
    if len({(r["epochs"], r["batch_size"]) for r in rows}) != 1:
        raise ValueError("Formal runs do not share one training budget and batch size")
    overall, subjects, seeds, deltas = aggregate(rows)
    for name, records in (("overall", overall), ("subjects", subjects),
                          ("seeds", seeds), ("paired_ablations", deltas)):
        write_csv(args.out / (name + ".csv"), records)
    if args.report:
        from eeg_report import build_report
        build_report(args, rows, overall, subjects, deltas)


if __name__ == "__main__":
    main()
