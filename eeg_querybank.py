"""Test whether training EEG predicts template false positives, then gate QB-Norm."""
import numpy as np
import torch
from scipy.stats import spearmanr

from eeg_class_alignment import MODEL, PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_train import DEVICE, OUT, load_json, model_for, run_dir, save_json, score, seed_run, tensor, write_log


BASE = OUT / "querybank"
GROUPS = {"pilot": ((1, 2), (0,)), "replication": (range(3, 11), (1, 2))}
PROTOCOL = dict(model=MODEL, checkpoint="development epoch 100", calibration="14540 development training EEGs",
                measure="Spearman correlation across 200 templates: training top-1 counts vs validation false top-1 counts",
                diagnostic_pilot="both rho >= 0.30",
                diagnostic_replication=">=6/8 subject-mean rho >=0.30, each seed mean rho >=0.30",
                concentration="fraction of validation errors in top 20 training templates, ties by class index",
                adjustment="dynamic inverted softmax, k=1, beta=20; reference training scores only",
                source="https://github.com/ioanacroi/qb-norm/blob/master/dynamic_inverted_softmax.py",
                retrieval_pilot=GATES["pilot"], retrieval_replication=GATES["replication"],
                official_test_used=False, new_fits=0)


def control(stage, sub, seed, design):
    path = (OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{sub:02d}.npz" if stage == "pilot"
            else OUT / "pooling_replication" / f"ab_pool4_sub-{sub:02d}_seed-{seed}.npz")
    with np.load(path) as data:
        np.testing.assert_array_equal(data["labels"], design["dev_labels"])
        np.testing.assert_array_equal(data["query_rows"], design["dev_query"])
        return data["scores"]


@torch.inference_mode()
def diagnose(stage, sub, seed, design, folder):
    seed_run(seed)
    ckpt = torch.load(run_dir("dev", MODEL, sub, seed) / "checkpoint.pt", map_location=DEVICE)
    settings = ckpt["settings"]
    assert (settings["subject"], settings["seed"], settings["config"], ckpt["trained_epochs"]) == (sub, seed, MODEL, 100)
    model = model_for(MODEL)
    model.load_state_dict(ckpt["model"])
    eeg, centers = load_subject(sub), tensor(design["dev_centers"])
    def scores(indices):
        return score(model, (tensor(eeg[indices]) - ckpt["mean"]) / ckpt["std"], centers, 128).cpu().numpy()

    bank = scores(design["dev_train"])
    raw = control(stage, sub, seed, design)
    np.testing.assert_allclose(scores(design["dev_query"]), raw, rtol=1e-5, atol=2e-6)
    assert bank.shape == (14540, 200) and np.isfinite(bank).all()
    counts = np.bincount(bank.argmax(1), minlength=200)
    predicted = raw.argmax(1)
    false = np.bincount(predicted[predicted != design["dev_labels"]], minlength=200)
    top = np.argsort(-counts, kind="stable")[:20]
    # Compact sufficient statistics for dynamic inverted softmax; labels and validation scores are excluded.
    log_z = np.log(np.exp(20 * bank.astype(np.float64)).sum(0))
    np.savez(folder / "template_counts.npz", training_top1=counts, validation_false_top1=false,
             log_denominator=log_z, template_ids=design["dev_classes"])
    return dict(subject=sub, seed=seed, rho=float(spearmanr(counts, false)[0]),
                top20_errors=int(false[top].sum()), total_errors=int(false.sum()),
                top20_error_share=float(false[top].sum()/false.sum()))


def diagnostic_gate(stage, runs):
    if stage == "pilot":
        return all(r["rho"] >= 0.30 for r in runs)
    subjects, seeds = GROUPS[stage]
    by_subject = [np.mean([r["rho"] for r in runs if r["subject"] == s]) for s in subjects]
    by_seed = [np.mean([r["rho"] for r in runs if r["seed"] == s]) for s in seeds]
    return sum(x >= 0.30 for x in by_subject) >= 6 and all(x >= 0.30 for x in by_seed)


def qb_scores(raw, stats):
    use = stats["training_top1"][raw.argmax(1)] > 0
    adjusted = raw.astype(np.float64)
    adjusted[use] = 20 * adjusted[use] - stats["log_denominator"]
    return adjusted, use


def main():
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as data:
        d = {k: data[k] for k in ("dev_train", "dev_query", "dev_labels", "dev_centers", "dev_classes")}
        assert np.unique(d["dev_train"]).size == len(d["dev_train"]) == 14540
        assert np.intersect1d(data["train_labels"][d["dev_train"]], d["dev_classes"]).size == 0
        np.testing.assert_array_equal(data["train_labels"][d["dev_query"]], d["dev_classes"][d["dev_labels"]])
    receipt = dict(official_test_used=False, new_fits=0, diagnostics=[], retrieval=[])
    for stage, (subjects, seeds) in GROUPS.items():
        runs = []
        for sub in subjects:
            for seed in seeds:
                folder = BASE / "diagnostics" / stage / f"sub-{sub:02d}" / f"seed-{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                runs.append(diagnose(stage, sub, seed, d, folder))
        write_log(BASE / "diagnostics" / stage / "runs.csv", runs)
        passed = bool(diagnostic_gate(stage, runs))
        receipt["diagnostics"].append(dict(stage=stage, runs=len(runs), passed=passed,
            mean_rho=float(np.mean([r["rho"] for r in runs]))))
        save_json(BASE / "summary.json", receipt)
        if not passed:
            return
    for stage, (subjects, seeds) in GROUPS.items():
        runs = []
        for sub in subjects:
            for seed in seeds:
                key = f"sub-{sub:02d}/seed-{seed}"
                raw = control(stage, sub, seed, d)
                with np.load(BASE / "diagnostics" / stage / key / "template_counts.npz") as stats:
                    adjusted, use = qb_scores(raw, stats)
                folder = BASE / "qb_norm" / stage / key
                folder.mkdir(parents=True, exist_ok=True)
                row = paired(sub, seed, folder, d["dev_query"], d["dev_labels"], adjusted, raw)
                row["adjusted_queries"] = int(use.sum())
                runs.append(row)
                np.savez_compressed(folder / "validation.npz", scores=adjusted, labels=d["dev_labels"],
                                    query_rows=d["dev_query"], ranks=ranks_and_accuracy(adjusted, d["dev_labels"])[0])
        result = summarize(stage, runs, BASE / "qb_norm")
        receipt["retrieval"].append(result)
        save_json(BASE / "summary.json", receipt)
        if not result["passed"]:
            return


if __name__ == "__main__":
    main()
