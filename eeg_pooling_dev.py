"""Test 4-point pooling against the original A+B development checkpoints."""
import numpy as np
import torch

from eeg_data import CACHE, load_subject
from eeg_evaluate import ranks_and_accuracy
from eeg_train import (DEVICE, OUT, model_for, read_log, run_dir, save_json, score,
                       seed_run, tensor, train, write_log)


def evaluate_checkpoint(config, subject, seed, design, eeg, centers, destination, stage="dev", *, epochs=100,
                        checkpoint_name="checkpoint.pt"):
    folder = run_dir(stage, config, subject, seed)
    ckpt = torch.load(folder / checkpoint_name, map_location=DEVICE)
    assert ckpt["trained_epochs"] == epochs
    model = model_for(config)
    model.load_state_dict(ckpt["model"], strict=True)
    scores = score(model, (eeg - ckpt["mean"]) / ckpt["std"], centers, 128).cpu().numpy()
    ranks, top1, top5 = ranks_and_accuracy(scores, design["dev_labels"])
    if checkpoint_name == "checkpoint.pt":
        recorded = read_log(folder / "train.csv")[-1]
        expected = (float(recorded["val_top1"]), float(recorded["val_top5"]))
    else:
        expected = ckpt["validation_accuracy"]
    np.testing.assert_allclose((top1, top5), expected, rtol=0, atol=1e-6)
    temp = destination.with_suffix(".tmp.npz")
    np.savez(temp, scores=scores, ranks=ranks, labels=design["dev_labels"], query_rows=design["dev_query"])
    temp.replace(destination)
    row = dict(config=config, subject=subject, seed=seed, epochs=epochs, top1_pct=top1*100,
               top5_pct=top5*100, parameters=sum(p.numel() for p in model.parameters()))
    return row, ckpt["mean"], ckpt["std"]


def main():
    output = OUT / "development_time_resolution"
    output.mkdir(exist_ok=True)
    save_json(output / "cnn_protocol.json", dict(
        subjects=[1, 2], seed=0, epochs=100, batch_size=128, controls="ab, existing development checkpoints",
        comparison="ab_pool4 minus ab on identical 1000 validation queries; direct scores from epoch 100",
        candidate_for_replication="Positive Top-1 difference on both subjects and positive mean Top-5 difference",
        split="official training partition only", score="unit EEG dot development class center",
    ))
    with np.load(CACHE / "design.npz") as file:
        design = {key: file[key] for key in ("train_images", "dev_train", "dev_query", "dev_centers", "dev_labels")}
    # Check the initialization independently of dropout before training.
    seed_run(0)
    control = model_for("ab").eval()
    seed_run(0)
    finer = model_for("ab_pool4").eval()
    with torch.no_grad():
        probe = tensor(load_subject(1, "training")[design["dev_train"][:8]])
        torch.testing.assert_close(finer(probe), control(probe), rtol=2e-5, atol=2e-6)
    del control, finer, probe
    rows = []
    for subject in (1, 2):
        train("ab_pool4", subject, 0, "dev", 100, 128, design)
        eeg = tensor(load_subject(subject, "training")[design["dev_query"]])
        centers = tensor(design["dev_centers"])
        train_mean = train_std = None
        for config in ("ab", "ab_pool4"):
            row, mu, std = evaluate_checkpoint(config, subject, 0, design, eeg, centers,
                                                output / f"cnn_{config}_sub-{subject:02d}.npz")
            if train_mean is not None:
                torch.testing.assert_close(mu, train_mean)
                torch.testing.assert_close(std, train_std)
            train_mean, train_std = mu, std
            rows.append(row)
            print(rows[-1], flush=True)
    write_log(output / "cnn_runs.csv", rows)


if __name__ == "__main__":
    main()
