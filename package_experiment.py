"""Validate 250 formal runs and package the experiment data or final submission."""
import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from eeg_evaluate import CONFIGS, aggregate, collect, ranks_and_accuracy


def archive_label(student_id, name):
    parts = (student_id or "待填写学号", name or "待填写姓名")
    if any(any(ch in '<>:"/\\|?*' or ord(ch) < 32 for ch in value)
           or value.strip(". ") != value or not value for value in parts):
        raise ValueError("Student ID and name must be valid filename components")
    return "+".join(parts)


def check_summary(path, records):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        actual = list(csv.DictReader(stream))
    error = f"Stale result table. Regenerate {path.name} with eeg_evaluate.py"
    if len(actual) != len(records):
        raise ValueError(error)
    for saved, expected in zip(actual, records):
        if saved.keys() != expected.keys():
            raise ValueError(error)
        for key, value in expected.items():
            if isinstance(value, float):
                if not math.isclose(float(saved[key]), value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(error)
            elif saved[key] != str(value):
                raise ValueError(error)


def paired_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {(int(r["subject"]), int(r["seed"])): r for r in rows}
    if len(rows) != 50 or set(indexed) != {(s, k) for s in range(1, 11) for k in range(5)}:
        raise ValueError("Expected all 50 unique subject/seed pairs in paired_runs.csv")
    return indexed


def report_files(evaluation, evidence_only):
    if evidence_only:
        return set()
    report = evaluation / "实验报告.docx"
    if not report.is_file():
        raise ValueError(f"Missing final Word report: {report}")
    with ZipFile(report) as doc:
        if "word/document.xml" not in doc.namelist() or doc.testzip():
            raise ValueError("Invalid Word report")
    figures = {p for p in (evaluation / "图表").rglob("*") if p.is_file()}
    if not any(p.suffix == ".png" for p in figures) or not any(p.suffix == ".csv" for p in figures):
        raise ValueError("Missing numbered report figures or source data")
    return figures | {report}


def followup_files(outputs, references):
    folder = outputs / "pooling_formal"
    table = folder / "paired_runs.csv"
    files = {table, folder / "summary.json", folder / "protocol.json"}
    paired = paired_rows(table)
    measured = []
    for key, r in sorted(paired.items()):
        subject, seed = key
        run = folder / "ab_pool4" / f"sub-{subject:02d}" / f"seed-{seed}"
        scores = np.load(run / "scores.npy", allow_pickle=False)
        if scores.shape != (200, 200):
            raise ValueError(f"Unexpected score matrix in {run}")
        _, top1, top5 = ranks_and_accuracy(scores, np.arange(200))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if (metrics["config"], metrics["subject"], metrics["seed"]) != ("ab_pool4", *key):
            raise ValueError(f"Wrong run metadata in {run}")
        old = references[key]
        expected = [round(100*v, 8) for v in (top1, top5, old["top1"], old["top5"])]
        expected += [expected[0] - expected[2], expected[1] - expected[3]]
        columns = ("top1_pct", "top5_pct", "old_top1_pct", "old_top5_pct", "top1_delta_pp", "top5_delta_pp")
        if (not np.allclose([float(r[k]) for k in columns], expected, atol=1e-6, rtol=0)
                or not np.allclose([metrics["top1"], metrics["top5"]], [top1, top5], atol=1e-6, rtol=0)):
            raise ValueError(f"Scores and paired table disagree for {run}")
        measured.append(expected[:2])
        with (run / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
            epochs = list(csv.DictReader(stream))
        if ([int(e["epoch"]) for e in epochs] != list(range(1, 101))
                or not all(np.isfinite(float(e["loss"])) for e in epochs)):
            raise ValueError(f"Incomplete training log in {run}")
        files.update(run / n for n in ("settings.json", "training.json", "checkpoint.pt", "metrics.json",
                                      "scores.npy", "train.csv"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["runs"] != 50 or not np.allclose(np.mean(measured, axis=0), summary["mean_pct"], rtol=0, atol=1e-6):
        raise ValueError("Follow-up scores disagree with summary.json")
    files.update(folder.glob("*.log"))
    files.update(folder.glob("*.err"))
    return files


def submission_files(root, evidence_only=False):
    outputs = root / "outputs"
    runs = outputs / "runs"
    rows, pending = collect(runs)
    if pending:
        raise ValueError(f"Verified {len(rows)}/200 runs; finish training and evaluation before packaging.")
    evaluation = outputs / "evaluation"
    pack = {
        root / name for name in (
            "eeg_data.py", "eeg_model.py", "eeg_train.py", "eeg_evaluate.py", "eeg_report.py",
            "eeg_pooling_formal.py", "eeg_pooling_dev.py", "eeg_pooling_replication.py",
            "eeg_time_resolution.py", "run_experiment.py", "verify_delivery.py", "build_delivery_figures.py",
            "package_experiment.py", "README_EXPERIMENT.md", "experiment_literature.csv",
            "requirements-experiment.txt", "experiment_environment.json",
            "_work/download_things_eeg2.py", "data/THINGS-EEG2/image_metadata.npy",
            "data/THINGS-EEG2/osf_manifest.json", "outputs/frozen.json", "outputs/batch.json",
            "data/THINGS-EEG2/derived/design.npz", "data/THINGS-EEG2/derived/sample_index.csv",
            "reports/figures/fig01_eeg_waveform.csv", "reports/figures/fig01_eeg_waveform.png",
            "reports/figures/sub01_waveform_input.npz",
            "reports/data_audit.json", "outputs/development_selection.csv", "outputs/pipeline.log",
            "outputs/debug/unicode-save.txt",
        )
    }
    pack.update((root / "tests").glob("test_*.py"))
    derived = root / "data" / "THINGS-EEG2" / "derived"
    for subject in range(1, 11):
        for split in ("training", "test"):
            pack.add(derived / f"sub-{subject:02d}_{split}.npy")
    for row in rows:
        folder = runs / row["config"] / f"sub-{row['subject']:02d}" / f"seed-{row['seed']}"
        for name in ("checkpoint.pt", "settings.json", "training.json", "metrics.json", "scores.npy", "train.csv"):
            pack.add(folder / name)
    for config in CONFIGS:
        for subject in (1, 2):
            pack.add(outputs / "dev" / config / f"sub-{subject:02d}" / "seed-0" / "train.csv")

    pack.update(report_files(evaluation, evidence_only))
    references = {(r["subject"], r["seed"]): r for r in rows if r["config"] == "ab"}
    pack.update(followup_files(outputs, references))
    evidence = outputs / "delivery_evidence"
    fig_files = list(evidence.glob("*.png"))
    if len(fig_files) < 5 or not all(p.with_suffix(".csv").exists() or p.with_suffix(".npy").exists()
                                     for p in fig_files):
        raise ValueError("Missing standalone delivery figures or source data")
    for path in ("dev", "smoke", "checks", "debug", "pooling_replication", "development_time_resolution", "delivery_evidence"):
        pack.update(p for p in (outputs / path).rglob("*") if p.is_file()
                    and p.suffix in {".pt", ".csv", ".json", ".npy", ".npz", ".txt", ".log", ".err", ".png", ".md"})
    for path in ("pooling_replication", "development_time_resolution"):
        pack.update(outputs / path / n for n in ("runs.csv", "protocol.json", "summary.json", "paired_differences.csv"))
    for subject in range(1, 11):
        for model in ("full80", "pooled10"):
            pack.update(outputs / "development_time_resolution" / model / f"sub-{subject:02d}" / n
                        for n in ("weights.npz", "scores.npy", "ranks.npy"))
    for name in ("runs", "pending", "overall", "subjects", "seeds", "paired_ablations"):
        pack.add(evaluation / (name + ".csv"))
    pack.add(evaluation / "status.json")
    missing = [str(p.relative_to(root)) for p in sorted(pack)
               if not p.is_file() or (p.stat().st_size == 0 and p.suffix != ".err")]
    if missing:
        raise ValueError("Missing submission materials: " + ", ".join(missing))

    frozen = json.loads((outputs / "frozen.json").read_text(encoding="utf-8"))
    if any(row["epochs"] != frozen["epochs"] or row["batch_size"] != frozen["batch_size"] for row in rows):
        raise ValueError("Run metadata disagrees with frozen training configuration")
    check_summary(evaluation / "runs.csv", rows)
    for name, summary in zip(("overall", "subjects", "seeds", "paired_ablations"), aggregate(rows)):
        check_summary(evaluation / (name + ".csv"), summary)
    return sorted(pack)


def create_archive(root, destination, label, evidence_only=False):
    files = submission_files(root, evidence_only)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (label + ".zip")
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".partial", delete=False) as stream:
        temp = Path(stream.name)
    try:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for path in files:
                archive.write(path, (Path(label) / path.relative_to(root)).as_posix())
        with ZipFile(temp) as archive:
            if archive.testzip():
                raise ValueError("Archive CRC verification failed")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evidence-only", action="store_true",
                        help="Package experiment evidence before the final report is available")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        label = "THINGS-EEG2_实验材料" if args.evidence_only else archive_label(args.student_id, args.name)
        if args.check_only:
            print(f"Checked 250 formal runs and {len(submission_files(root, args.evidence_only))} files.")
        else:
            print(create_archive(root, args.destination or root / "dist", label, args.evidence_only))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
