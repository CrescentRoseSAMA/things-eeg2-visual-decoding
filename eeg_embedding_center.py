"""Score frozen EEG encoders after subtracting the training embedding mean."""
import numpy as np
import torch
import torch.nn.functional as F

from eeg_class_alignment import MODEL, PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_train import DEVICE, OUT, load_json, model_for, run_dir, save_json, seed_run, tensor


BASE = OUT / "embedding_center"
PROTOCOL = dict(model=MODEL, checkpoint="dev, epoch 100", mean_rows="all 14540 dev_train rows only",
                query="1000 held-out queries; unchanged 200 five-image template means",
                score="L2-normalize(query embedding minus training embedding mean) dot raw templates",
                coefficient=1, training_runs=0, pilot=GATES["pilot"], replication=GATES["replication"],
                transitions=["rank 2..5 to 1", "rank >5 to 1", "rank 1 to >1"], formal_test_used=False)


@torch.inference_mode()
def evaluate(stage, sub, seed, design):
    seed_run(seed)
    checkpoint = torch.load(run_dir("dev", MODEL, sub, seed) / "checkpoint.pt", map_location=DEVICE)
    settings = checkpoint["settings"]
    assert (settings["config"], settings["subject"], settings["seed"], checkpoint["trained_epochs"]) == (
        MODEL, sub, seed, 100)
    model = model_for(MODEL).eval()
    model.load_state_dict(checkpoint["model"])
    eeg = load_subject(sub, "training")
    def encode(indices):
        x = (tensor(eeg[indices]) - checkpoint["mean"]) / checkpoint["std"]
        return torch.cat([model(batch) for batch in x.split(128)])

    mu = encode(design["dev_train"]).mean(dim=0)
    rows, labels = design["dev_query"], design["dev_labels"]
    queries, centers = encode(rows), tensor(design["dev_centers"])
    raw = (queries @ centers.T).cpu().numpy()
    reference = (OUT / "development_time_resolution" / f"cnn_ab_pool4_sub-{sub:02d}.npz" if stage == "pilot"
                 else OUT / "pooling_replication" / f"ab_pool4_sub-{sub:02d}_seed-{seed}.npz")
    with np.load(reference) as saved:
        np.testing.assert_array_equal(saved["query_rows"], rows)
        np.testing.assert_array_equal(saved["labels"], labels)
        np.testing.assert_allclose(raw, saved["scores"], atol=2e-6, rtol=1e-5)
        old = saved["ranks"]
    np.testing.assert_array_equal(old, ranks_and_accuracy(raw, labels)[0])
    adjusted = (F.normalize(queries - mu, dim=1) @ centers.T).cpu().numpy()
    ranks = ranks_and_accuracy(adjusted, labels)[0]
    folder = BASE / stage / f"sub-{sub:02d}" / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "validation.npz", scores=adjusted, ranks=ranks, labels=labels,
                        query_rows=rows, training_embedding_mean=mu.cpu().numpy())
    row = paired(sub, seed, folder, rows, labels, adjusted, raw)
    row.update(rank2to5_to_first=int(((old >= 2) & (old <= 5) & (ranks == 1)).sum()),
               below5_to_first=int(((old > 5) & (ranks == 1)).sum()),
               lost_first=int(((old == 1) & (ranks > 1)).sum()))
    assert row["gain1"] == row["rank2to5_to_first"] + row["below5_to_first"] - row["lost_first"]
    return row


def main():
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        design = {k: file[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels")}
        idx = design["dev_train"]
        assert len(np.unique(idx)) == len(idx) == 14540
        assert not np.intersect1d(file["train_labels"][idx], file["dev_classes"]).size
        assert design["dev_centers"].shape == (200, 768) and len(design["dev_query"]) == 1000
    stages = []
    for stage, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        runs = [evaluate(stage, sub, seed, design) for sub in subjects for seed in seeds]
        stages.append(summarize(stage, runs, BASE))
        if not stages[-1]["passed"]:
            break
    save_json(BASE / "summary.json", dict(stopped_at=stage, formal_test_used=False, stages=stages))


if __name__ == "__main__":
    main()
