"""Rerank the five QB-Norm candidates using the original EEG/image dot products."""
import numpy as np

from eeg_class_alignment import PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE
from eeg_querybank import BASE as BANK, GROUPS, control, qb_scores
from eeg_train import OUT, load_json, save_json


BASE = OUT / "qb_shortlist"
PROTOCOL = dict(control="ab_pool4 epoch 100", candidates="top 5 from frozen QB-Norm, beta 20, k 1",
                rerank="original ab_pool4 score within the five; retain QB order below rank five",
                ties="ascending template index", pilot=GATES["pilot"], replication=GATES["replication"],
                new_fits=0, official_test_used=False)


def main():
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing shortlist protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        design = {k: file[k] for k in ("dev_query", "dev_labels", "dev_classes")}
    labels = design["dev_labels"]
    stages = []
    for stage, (subjects, seeds) in GROUPS.items():
        runs = []
        for sub in subjects:
            for seed in seeds:
                key = f"sub-{sub:02d}/seed-{seed}"
                raw = control(stage, sub, seed, design)
                with np.load(BANK / "diagnostics" / stage / key / "template_counts.npz") as stats:
                    np.testing.assert_array_equal(stats["template_ids"], design["dev_classes"])
                    adjusted, _ = qb_scores(raw, stats)
                if stage == "pilot":
                    with np.load(BANK / "qb_norm" / stage / key / "validation.npz") as saved:
                        np.testing.assert_array_equal(adjusted, saved["scores"])
                order = np.argsort(-adjusted, axis=1, kind="stable")
                ids = np.sort(order[:, :5], axis=1)
                within = np.take_along_axis(raw, ids, axis=1)
                order[:, :5] = np.take_along_axis(ids, np.argsort(-within, axis=1, kind="stable"), axis=1)
                # Convert full candidate order to ordinal scores for the shared ranking evaluator.
                ordinal = np.empty_like(order)
                np.put_along_axis(ordinal, order, -np.arange(raw.shape[1])[None, :], axis=1)
                folder = BASE / stage / key
                folder.mkdir(parents=True, exist_ok=True)
                row = paired(sub, seed, folder, design["dev_query"], labels, ordinal, raw)
                old, new = raw.argmax(1), order[:, 0]
                row.update(changed_first=int((old != new).sum()),
                           gained_first=int(((old != labels) & (new == labels)).sum()),
                           lost_first=int(((old == labels) & (new != labels)).sum()))
                assert row["gain1"] == row["gained_first"] - row["lost_first"]
                assert row["top5_hits"] == int((ids == labels[:, None]).any(1).sum())
                np.savez_compressed(folder / "rankings.npz", candidate_order=order, labels=labels,
                                    query_rows=design["dev_query"])
                runs.append(row)
        stages.append(summarize(stage, runs, BASE))
        save_json(BASE / "summary.json", dict(stages=stages, new_fits=0, official_test_used=False))
        if not stages[-1]["passed"]:
            break


if __name__ == "__main__":
    main()
