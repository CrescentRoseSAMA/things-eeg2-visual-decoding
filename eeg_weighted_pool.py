"""Learn the four within-window pooling weights on the development split."""
import numpy as np
import torch

from eeg_class_alignment import MODEL, PROTOCOL as GATES, development
from eeg_data import CACHE, load_subject
from eeg_model import contrastive_loss
from eeg_train import DEVICE, OUT, load_json, model_for, run_dir, save_json, seed_run, tensor, write_log


BASE = OUT / "weighted_pool"
CANDIDATE = "ab_pool4_weighted"
PROTOCOL = dict(control=MODEL, candidate=CANDIDATE, pooling="32 channel-specific softmax distributions "
                "over four consecutive samples, stride four, shared across windows, all logits start at zero",
                extra_parameters=128, loss="symmetric image InfoNCE", lr=0.0002, temperature=0.07,
                batch=128, epochs=100, checkpoint_interval=5, pilot=GATES["pilot"],
                replication=GATES["replication"], official_test_used=False)


def verify_initialization(design):
    ids = design["dev_train"][:128]
    ref = torch.load(run_dir("dev", MODEL, 1, 0) / "checkpoint.pt", map_location=DEVICE)
    x = (tensor(load_subject(1)[ids]) - ref["mean"]) / ref["std"]
    seed_run(0)
    fixed = model_for(MODEL).eval()
    seed_run(0)
    candidate = model_for(CANDIDATE).eval()
    for name, value in fixed.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, atol=0, rtol=0)
    count = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in fixed.parameters())
    assert count == 128
    pool = candidate.readout[3]
    torch.testing.assert_close(pool.logits.softmax(dim=1), torch.full_like(pool.logits, 0.25), atol=0, rtol=0)
    with torch.no_grad():
        original = fixed(x)
    output = candidate(x)
    torch.testing.assert_close(output, original, atol=2e-6, rtol=2e-5)
    contrastive_loss(output, tensor(design["train_images"][ids])).backward()
    gradient = pool.logits.grad
    assert gradient.isfinite().all() and torch.count_nonzero(gradient).item() == count
    save_json(BASE / "initialization.json", dict(probe_rows=len(ids), added_parameters=count,
              nonzero_gradients=count, max_output_difference=(original-output).abs().max().item()))


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    BASE.mkdir(exist_ok=True)
    manifest = BASE / "protocol.json"
    if manifest.exists() and load_json(manifest) != PROTOCOL:
        raise ValueError("Existing protocol differs")
    save_json(manifest, PROTOCOL)
    with np.load(CACHE / "design.npz") as file:
        design = {k: file[k] for k in ("dev_train", "dev_query", "dev_centers", "dev_labels",
                                       "train_images", "train_labels")}
    verify_initialization(design)
    stages = []
    for stage, subjects, seeds in (("pilot", (1, 2), (0,)), ("replication", range(3, 11), (1, 2))):
        stages.append(development(stage, subjects, seeds, design, BASE, config=CANDIDATE, class_loss=False))
        for sub in subjects:
            for seed in seeds:
                folder = run_dir(f"{BASE.name}/{stage}", CANDIDATE, sub, seed)
                ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu")
                weights = ckpt["model"]["readout.3.logits"].softmax(dim=1).numpy()
                np.testing.assert_allclose(weights.sum(1), 1, atol=2e-7)
                assert np.any(weights != 0.25)
                write_log(folder / "pool_weights.csv", [
                    dict(channel=i+1, **{f"position_{j+1}": float(v) for j, v in enumerate(row)})
                    for i, row in enumerate(weights)])
        save_json(BASE / "summary.json", dict(last_stage=stage, stages=stages, official_test_used=False))
        if not stages[-1]["passed"]:
            break


if __name__ == "__main__":
    main()
