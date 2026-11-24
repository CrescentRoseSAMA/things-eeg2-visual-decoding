"""Paired development evaluation of a fixed EEG CNN and full-resolution ridge blend."""
import numpy as np

from eeg_class_alignment import PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE
from eeg_evaluate import ranks_and_accuracy
from eeg_train import OUT, load_json, save_json


BASE = OUT / "ridge_fusion"
PROTOCOL = dict(cnn="ab_pool4, epoch 100", ridge="full80, alpha 0.1, same subject and split",
                score="0.75 * original CNN dot product + 0.25 * full80 ridge dot product",
                weight_search=False, query_count=1000, pilot=GATES["pilot"],
                replication=GATES["replication"], training_runs=0, official_test_used=False)


def main():
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing fusion protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as design:
        idx, labels = design["dev_query"], design["dev_labels"]
    stages = []
    for stage, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        runs = []
        for sub in subjects:
            key = f"sub-{sub:02d}"
            time_dir = OUT / "development_time_resolution"
            ridge_dir = time_dir / "full80" / key
            ridge = np.load(ridge_dir / "scores.npy")
            assert ridge.shape == (1000, 200)
            ridge_rank = ranks_and_accuracy(ridge, labels)[0]
            np.testing.assert_array_equal(ridge_rank, np.load(ridge_dir / "ranks.npy"))
            for seed in seeds:
                path = (time_dir / f"cnn_ab_pool4_{key}.npz" if stage == "pilot" else
                        OUT / "pooling_replication" / f"ab_pool4_{key}_seed-{seed}.npz")
                with np.load(path) as control:
                    np.testing.assert_array_equal(control["query_rows"], idx)
                    np.testing.assert_array_equal(control["labels"], labels)
                    raw, old_rank = control["scores"], control["ranks"]
                np.testing.assert_array_equal(old_rank, ranks_and_accuracy(raw, labels)[0])
                fused = 0.75 * raw.astype(np.float64) + 0.25 * ridge
                folder = BASE / stage / key / f"seed-{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                result = paired(sub, seed, folder, idx, labels, fused, raw)
                rank = ranks_and_accuracy(fused, labels)[0]
                result.update(ridge_only_top1=int(((ridge_rank == 1) & (old_rank > 1)).sum()),
                              gained_top1=int(((rank == 1) & (old_rank > 1)).sum()),
                              lost_top1=int(((rank > 1) & (old_rank == 1)).sum()))
                runs.append(result)
                np.savez_compressed(folder / "validation.npz", scores=fused, labels=labels,
                                    ranks=rank, query_rows=idx)
        stages.append(summarize(stage, runs, BASE))
        if not stages[-1]["passed"]:
            break
    save_json(BASE / "summary.json", dict(stages=stages, training_runs=0, official_test_used=False))


if __name__ == "__main__":
    main()
