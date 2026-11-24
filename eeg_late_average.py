"""Compare an equal-weight parameter average against the final model on its own training trajectory."""
import numpy as np
import torch

from eeg_class_alignment import MODEL, PROTOCOL as GATES, paired, summarize
from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_pooling_dev import evaluate_checkpoint
from eeg_train import DEVICE, OUT, load_json, model_for, run_dir, save_json, save_weights, score, tensor, train


BASE = OUT / "late_average"
EPOCHS = (80, 85, 90, 95, 100)
PROTOCOL = dict(model=MODEL, epochs=list(EPOCHS), weights=[0.2]*5, fit_epochs=100, adam_lr=0.0002,
                batch_size=128, temperature=0.07, loss="symmetric image InfoNCE",
                control="same training trajectory, epoch 100", normalization="identical epoch 100 "
                "batch normalization buffers and identical input statistics for both models",
                pilot=GATES["pilot"], replication=GATES["replication"], official_test_used=False)


@torch.no_grad()
def average_and_score(folder, x, centers):
    final = torch.load(folder / "checkpoint.pt", map_location="cpu")
    model = model_for(MODEL)
    model.load_state_dict(final["model"])
    snapshots = [torch.load(folder / f"epoch-{e}.pt", map_location="cpu") for e in EPOCHS[:-1]]
    for name, param in model.named_parameters():
        values = [state[name] for state in snapshots] + [final["model"][name]]
        param.copy_(torch.stack(values).mean(0))
    saved = folder / "average.pt"
    save_weights(saved, dict(model=model.state_dict(), mean=final["mean"], std=final["std"],
                             settings=final["settings"], averaged_epochs=list(EPOCHS)))
    checkpoint = torch.load(saved, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    return score(model, (x-checkpoint["mean"])/checkpoint["std"], centers, 128).cpu().numpy()


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing averaging protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as f:
        d = {k: f[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels", "train_images")}
        assert not np.intersect1d(f["train_labels"][d["dev_train"]], f["dev_classes"]).size
    rows, labels = d["dev_query"], d["dev_labels"]
    centers = tensor(d["dev_centers"])
    assert (len(d["dev_train"]), len(rows), len(centers)) == (14540, 1000, 200)
    stages = []
    for name, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        runs = []
        stage = f"{BASE.name}/{name}"
        for sub in subjects:
            x = tensor(load_subject(sub, "training")[rows])
            for seed in seeds:
                train(MODEL, sub, seed, stage, 100, 128, d, split="dev", snapshot_epochs=EPOCHS[:-1])
                folder = run_dir(stage, MODEL, sub, seed)
                path = folder / "epoch100_validation.npz"
                evaluate_checkpoint(MODEL, sub, seed, d, x, centers, path, stage)
                with np.load(path) as endpoint:
                    old = endpoint["scores"]
                new = average_and_score(folder, x, centers)
                ranks = ranks_and_accuracy(new, labels)[0]
                np.savez_compressed(folder / "average_validation.npz", scores=new, ranks=ranks,
                                    labels=labels, query_rows=rows)
                runs.append(paired(sub, seed, folder, rows, labels, new, old))
        stages.append(summarize(name, runs, BASE))
        save_json(BASE / "summary.json", dict(stages=stages, official_test_used=False))
        if not stages[-1]["passed"]:
            break


if __name__ == "__main__":
    main()
