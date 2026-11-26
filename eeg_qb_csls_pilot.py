"""Compare fixed QB-Norm and CSLS on the same round 06 checkpoints."""
import argparse
from itertools import product

import numpy as np
import torch

from eeg_csls_replication import CANDIDATE, K, csls_scores, training_stats
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_gemini_iterations import BASE, BATCH, COURSE, PILOT, design_for, freeze, pairs_as_lists, read_development, result_row
from eeg_pooling_formal import checked_metrics
from eeg_querybank import BETA, qb_scores, querybank_statistics
from eeg_repeat_views import TRIAL_SAMPLING
from eeg_train import DEVICE, OUT, load_json, model_for, run_dir, score, seed_run, tensor
from eeg_trial_augmentation import OUTPUT, gate, stored_csls, summarize


DIRECTORY = BASE / "round-13"
CANDIDATE_ARM = "round06_qb_norm"
CONTROL_ARM = "round06_csls"


def main(*, full=False):
    phase = "full" if full else "pilot"
    directory = DIRECTORY / phase
    stage = (OUTPUT / ("formal" if full else "pilot")).relative_to(OUT).as_posix()
    pairs = tuple(product(range(1, 11), range(5))) if full else PILOT
    protocol = dict(
        model=CANDIDATE, candidate=CANDIDATE_ARM, control=CONTROL_ARM, checkpoint_stage=stage,
        pairs=pairs_as_lists(pairs),
        gallery="identical 200 frozen CLIP templates normalized for both methods",
        qb_beta=BETA, qb_k=1, csls_k=K, query_statistics="independent rescoring per query",
        new_fits=0, official_test_used=full)
    if full:
        pilot = load_json(DIRECTORY / "pilot" / "summary.json")
        protocol.update(calibration="same subject's 16540 standard four-repeat official training EEG averages",
                        query="identical 200 official eighty-repeat EEG averages per subject and seed",
                        selection="requested full-set follow-up with the scoring rule and beta fixed from the pilot",
                        pilot_gate_pass=pilot["gate_pass"], course_target_pct=list(COURSE))
        with np.load(CACHE / "design.npz") as cache:
            templates, labels = (cache[k] for k in ("test_centers", "test_labels"))
        train_rows, queries = np.arange(16540), np.arange(200)
        np.testing.assert_array_equal(labels, queries)
    else:
        protocol.update(split_seed=2026, calibration="same subject's standard four-repeat dev_train EEG",
                        query="identical 1000 four-repeat development EEG queries",
                        gate="Top-5 >= +1pp and Top-1 >= 0pp; at least 3/4 subjects improve Top-5")
        design = design_for(2026)
        train_rows, queries, templates, labels = (design[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels"))
    freeze(directory / "protocol.json", protocol)
    norms = np.linalg.norm(templates, axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError("Expected finite nonzero gallery templates")
    centers = tensor(templates)
    rows = []
    for subject, seed in pairs:
        seed_run(seed)
        reference = run_dir(stage, CANDIDATE, subject, seed)
        stats = training_stats(reference, CANDIDATE, subject, seed)
        if stats.get("training_eeg") != TRIAL_SAMPLING:
            raise ValueError(f"Expected round 06 trial augmentation: {reference}")
        if full:
            checked_metrics(reference)
            if stats["split"] != "runs":
                raise ValueError(f"Expected weights fitted on the official training classes: {reference}")
        else:
            read_development(reference / "validation.npz", design)
        old = stored_csls(reference / "csls.npz", labels)
        checkpoint = torch.load(reference / "checkpoint.pt", map_location=DEVICE)
        model = model_for(CANDIDATE)
        model.load_state_dict(checkpoint["model"], strict=True)
        eeg = load_subject(subject, "training")
        query_eeg = load_subject(subject, "test") if full else eeg[queries]
        mean, std = checkpoint["mean"], checkpoint["std"]
        bank = score(model, (tensor(eeg[train_rows]) - mean) / std, centers, BATCH).cpu().numpy() / norms
        raw = score(model, (tensor(query_eeg) - mean) / std, centers, BATCH).cpu().numpy() / norms
        if bank.shape != (len(train_rows), len(templates)) or not np.isfinite(bank).all():
            raise ValueError(f"Invalid calibration bank: {reference}")
        with np.load(reference / "csls.npz") as saved:
            np.testing.assert_allclose(raw, saved["normalized_scores"], atol=2e-6, rtol=1e-5)
            rebuilt_csls = csls_scores(raw, bank)
            np.testing.assert_allclose(rebuilt_csls, saved["csls_scores"], atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(ranks_and_accuracy(rebuilt_csls, labels)[0], old)
        bank_stats = querybank_statistics(bank)
        adjusted, used = qb_scores(raw, bank_stats)
        new = ranks_and_accuracy(adjusted, labels)[0]
        folder = directory / f"sub-{subject:02d}" / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "qb_scores.npz", query_rows=queries, train_rows=train_rows, labels=labels,
                            scores=adjusted, ranks=new, bank_cosine=bank, normalized_scores=raw, **bank_stats)
        row = result_row(subject, seed, CANDIDATE_ARM, CONTROL_ARM, new, old,
                         {CANDIDATE_ARM: stats, CONTROL_ARM: stats})
        row.update(adjusted_queries=int(used.sum()), top5_recovered=int(((new <= 5) & (old > 5)).sum()),
                   top5_lost=int(((old <= 5) & (new > 5)).sum()))
        rows.append(row)
    status = dict(new_fits=0, official_test_used=full)
    if full:
        by_seed = np.array([[np.mean([r[f"top{k}_pct"] for r in rows if r["seed"] == seed]) for k in (1, 5)]
                            for seed in range(5)])
        status.update(pilot_gate_pass=protocol["pilot_gate_pass"], seed_sd_pp=by_seed.std(0, ddof=1).tolist(),
                      delta_vs_course_pp=(by_seed.mean(0) - COURSE).tolist())
    else:
        status["gate_pass"] = gate("pilot", rows)
    summarize(directory, rows, **status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="score all 50 round 06 official-test pairs")
    main(full=parser.parse_args().full)
