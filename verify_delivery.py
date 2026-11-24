"""Replay every formal checkpoint against the EEG and class templates in a package."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg_evaluate import ranks_and_accuracy
from eeg_model import EEGEncoder


@torch.inference_mode()
def verify(root, device):
    cache = root / "data/THINGS-EEG2/derived"
    with np.load(cache / "design.npz") as design:
        labels = design["test_labels"]
        np.testing.assert_array_equal(labels, np.arange(200))
        centers = torch.tensor(design["test_centers"], device=device)
    programs = [(c, "runs") for c in ("baseline", "a", "b", "ab")]
    programs.append(("ab_pool4", "pooling_formal"))
    results = {c: [] for c, _ in programs}
    max_diff = 0.0
    for subject in range(1, 11):
        eeg = torch.tensor(np.load(cache / f"sub-{subject:02d}_test.npy"), device=device)
        for config, stage in programs:
            model = EEGEncoder(config).to(device).eval()
            for seed in range(5):
                run = root / "outputs" / stage / config / f"sub-{subject:02d}" / f"seed-{seed}"
                checkpoint = torch.load(run / "checkpoint.pt", map_location=device)
                settings = checkpoint["settings"]
                if (settings["subject"], settings["config"], settings["seed"]) != (subject, config, seed):
                    raise ValueError(f"Checkpoint identity mismatch: {run}")
                model.load_state_dict(checkpoint["model"], strict=True)
                x = (eeg - checkpoint["mean"]) / checkpoint["std"]
                scores = torch.cat([model(batch) @ centers.T for batch in x.split(settings["batch_size"])]).cpu().numpy()
                stored = np.load(run / "scores.npy")
                np.testing.assert_allclose(scores, stored, atol=2e-4, rtol=1e-4, err_msg=str(run))
                max_diff = max(max_diff, float(np.max(np.abs(scores - stored))))
                _, top1, top5 = ranks_and_accuracy(scores, labels)
                recorded = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                np.testing.assert_allclose([top1, top5], [recorded["top1"], recorded["top5"]],
                                           atol=1e-6, rtol=0, err_msg=str(run))
                results[config].append([100*top1, 100*top5])
    return {"device": str(device), "weights_replayed": sum(map(len, results.values())),
            "queries_replayed": 50000, "max_abs_score_error": max_diff,
            "mean_top1_top5_pct": {k: np.mean(v, axis=0).round(5).tolist() for k, v in results.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--receipt", type=Path, help="Optional path for the JSON verification result")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = json.dumps(verify(args.root.resolve(), torch.device(args.device)), indent=2)
    if args.receipt:
        args.receipt.write_text(result + "\n", encoding="utf-8")
    print(result)
