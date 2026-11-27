"""Train and reload the within-subject THINGS-EEG2 models on a single device."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

from eeg_data import load_design, load_subject
from eeg_model import (COSINE_REGRESSION_CONFIGS, COSINE_REGRESSION_WEIGHT, EEGPROJECT_CONFIG,
                       LEARNABLE_TEMPERATURE_CONFIG, MAX_LOGIT_SCALE, PAIR_TEMPERATURE,
                       SOFT_CLIP_TEMPERATURE, SOFT_CLIP_WEIGHT, ClassPrototypes, EEGEncoder, EEGProjectEncoder,
                       contrastive_loss, contrastive_temperature)
from eeg_repeat_views import (SINGLE_TRIAL_SAMPLING, TRIAL_CHANNEL_DROPOUT, TRIAL_NOISE_STD, TRIAL_SAMPLING,
                              perturb_trial_epoch, sample_epoch, sample_subset_epoch)


CONFIGS = ("baseline", "a", "b", "ab")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002
EMA_DECAY = 0.90
EMA_FIRST_EPOCH = 51
TOP5_RANK_WEIGHT = 0.10
TOP5_GALLERY_SIZE = 200


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_weights(path, state):
    tmp = path.with_suffix(".tmp")
    # The PyTorch 2.0 Windows filename writer fails for some Unicode paths.
    with tmp.open("wb") as stream:
        torch.save(state, stream)
    tmp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def tensor(array):
    return torch.as_tensor(np.array(array, dtype=np.float32, copy=True), device=DEVICE)


def run_dir(stage, config, subject, seed):
    return OUT / stage / config / f"sub-{subject:02d}" / f"seed-{seed}"


def model_for(config):
    model = EEGProjectEncoder() if config == EEGPROJECT_CONFIG else EEGEncoder(config)
    return model.to(DEVICE)


@torch.no_grad()
def score(model, x, centers, batch):
    model.eval()
    return torch.cat([model(chunk) @ centers.T for chunk in x.split(batch)])


def accuracy(scores, labels):
    # Stable ordering resolves equal scores by candidate index.
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :5]
    hits = order.eq(labels[:, None])
    return hits[:, 0].float().mean().item(), hits.any(1).float().mean().item()


def get_arrays(subject, design, stage):
    eeg = tensor(load_subject(subject, "training"))
    images = tensor(design["train_images"])
    if stage == "smoke":
        _, idx = np.unique(design["train_labels"], return_index=True)
        idx = idx[:32]
        train, images = eeg[idx], images[idx]
        val, centers, labels = train, images, np.arange(32)
    elif stage == "dev":
        train_idx, val_idx = design["dev_train"], design["dev_query"]
        train, images = eeg[train_idx], images[train_idx]
        val = eeg[val_idx]
        centers, labels = tensor(design["dev_centers"]), design["dev_labels"]
    else:
        train = eeg
        val = centers = labels = None
    mean = train.mean(dim=(0, 2), keepdim=True)
    std = train.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
    if val is not None:
        val = (val - mean) / std
        labels = torch.as_tensor(np.array(labels), dtype=torch.long, device=DEVICE)
    return (train - mean) / std, images, val, centers, labels, mean, std


def read_log(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_log(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def source_arrays(subjects, design):
    ids = design["dev_train"]
    signals, means, stds = [], [], []
    for subject in subjects:
        values = tensor(load_subject(subject, "training")[ids])
        mean = values.mean((0, 2), keepdim=True)
        std = values.std((0, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
        signals.append(values.sub_(mean).div_(std))
        means.append(mean)
        stds.append(std)
    return torch.stack(signals), tensor(design["train_images"][ids]), torch.stack(means), torch.stack(stds)


def source_indices(n_images, n_sources, seed, epoch):
    block, offset = divmod(epoch - 1, n_sources)
    rng = np.random.default_rng([seed, block, 651])
    return (rng.permutation(n_images) + offset) % n_sources


def make_weight_ema(model):
    return AveragedModel(model, avg_fn=lambda average, current, _: average.lerp(current, 1 - EMA_DECAY))


def train(config, subject, seed, stage, epochs, batch, design, *, split=None, class_labels=None,
          repeat_views=None, trial_repeats=None, sources=None, initial_checkpoint=None, snapshot_epochs=(),
          negatives=None, soft_clip=False, lr=LR, single_trials=False, perturb_trials=False, weight_ema=False,
          trial_noise_std=TRIAL_NOISE_STD, top5_ranking=False):
    if trial_repeats is not None and (repeat_views is not None or sources is not None):
        raise ValueError("Trial subset augmentation requires within-subject raw trials only")
    if (single_trials or perturb_trials) and trial_repeats is None:
        raise ValueError("Trial regularization requires raw trials")
    if weight_ema and epochs < EMA_FIRST_EPOCH:
        raise ValueError("The training budget ends before weight averaging starts")
    if top5_ranking and class_labels is None:
        raise ValueError("Top-5 ranking requires training class labels")
    folder = run_dir(stage, config, subject, seed)
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(config=config, subject=subject, seed=seed, stage=stage,
                    epochs=epochs, batch_size=batch, lr=lr, temperature=PAIR_TEMPERATURE)
    if config == LEARNABLE_TEMPERATURE_CONFIG:
        settings.update(learnable_temperature=True, max_logit_scale=MAX_LOGIT_SCALE)
    cosine_weight = COSINE_REGRESSION_WEIGHT if config in COSINE_REGRESSION_CONFIGS else 0.0
    if cosine_weight:
        settings["cosine_regression_weight"] = cosine_weight
    if soft_clip:
        settings.update(soft_clip_weight=SOFT_CLIP_WEIGHT, soft_clip_temperature=SOFT_CLIP_TEMPERATURE)
    if split is not None:
        settings["split"] = split
    if snapshot_epochs:
        settings["snapshot_epochs"] = list(snapshot_epochs)
    if negatives is not None:
        settings.update(extra_images=16, extra_sampling=negatives.mode)
    if sources is not None:
        settings["pretraining_subjects"] = list(sources)
    if initial_checkpoint is not None:
        settings["initial_checkpoint"] = initial_checkpoint.relative_to(OUT).as_posix()
    if repeat_views is not None:
        settings["training_eeg"] = "0.5 four-repeat mean, 0.5 uniform three-repeat mean"
    if trial_repeats is not None:
        settings["training_eeg"] = SINGLE_TRIAL_SAMPLING if single_trials else TRIAL_SAMPLING
    if perturb_trials:
        settings.update(trial_noise_std=trial_noise_std, trial_channel_dropout=TRIAL_CHANNEL_DROPOUT)
    if weight_ema:
        settings.update(ema_decay=EMA_DECAY, ema_first_epoch=EMA_FIRST_EPOCH)
    if top5_ranking:
        settings.update(top5_rank_weight=TOP5_RANK_WEIGHT, top5_gallery_size=TOP5_GALLERY_SIZE)
    elif class_labels is not None:
        settings.update(image_loss_weight=0.75, leave_one_out_class_loss_weight=0.25)
    if (folder / "settings.json").exists() and load_json(folder / "settings.json") != settings:
        raise ValueError(f"Existing run has different settings: {folder}")
    save_json(folder / "settings.json", settings)
    if (folder / "training.json").exists():
        return
    seed_run(seed)
    if sources is None:
        x, images, val, centers, labels, mean, std = get_arrays(subject, design, split or stage)
    else:
        source_x, images, mean, std = source_arrays(sources, design)
        x = source_x[0]
        image_rows = torch.arange(len(x), device=DEVICE)
        val = None
    if trial_repeats is not None:
        if trial_repeats.shape != (len(x), 4, *x.shape[1:]):
            raise ValueError("Expected four raw trials for each training EEG row")
        repeat_views = tensor(trial_repeats).sub_(mean).div_(std)
    elif repeat_views is not None:
        repeat_views = torch.as_tensor(repeat_views, device=DEVICE).sub(mean).div_(std)
    bank = ClassPrototypes(images, class_labels) if class_labels is not None else None
    if top5_ranking and len(bank.centers) < TOP5_GALLERY_SIZE:
        raise ValueError("Not enough training classes for the Top-5 gallery")
    if negatives is not None:
        negative_images = torch.nn.functional.normalize(images, dim=1)
    model = model_for(config)
    if initial_checkpoint is not None:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=DEVICE)["model"], strict=True)
    ema = make_weight_ema(model) if weight_ema else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    if top5_ranking:
        ranking_generator = torch.Generator(device=DEVICE)
    resume = folder / "resume.pt"
    log_path = folder / "train.csv"
    rows = []
    if resume.exists():
        state = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(state["model"], strict=True)
        if ema is not None:
            ema.load_state_dict(state["ema"], strict=True)
        opt.load_state_dict(state["optimizer"])
        generator.set_state(state["shuffle"].cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        rows = read_log(log_path)[:state["epoch"]]
        if len(rows) != state["epoch"]:
            raise ValueError(f"Incomplete training log: {folder}")
        del state
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"{stage} {config} sub-{subject:02d} seed-{seed}: epoch {len(rows)+1}/{epochs}", flush=True)
    for epoch in range(len(rows) + 1, epochs + 1):
        start = time.perf_counter()
        model.train()
        losses = torch.zeros((), device=DEVICE)
        epoch_x = x
        if trial_repeats is not None:
            epoch_x = sample_subset_epoch(x, repeat_views, seed, epoch, min_trials=1 if single_trials else 2)
            if perturb_trials:
                epoch_x = perturb_trial_epoch(epoch_x, seed, epoch, noise_std=trial_noise_std)
        elif repeat_views is not None:
            epoch_x = sample_epoch(x, repeat_views, seed, epoch)
        if sources is not None and len(sources) > 1:
            selected = torch.as_tensor(source_indices(len(x), len(sources), seed, epoch), device=DEVICE)
            epoch_x = source_x[selected, image_rows]
        order = torch.randperm(len(x), generator=generator, device=DEVICE)
        if top5_ranking:
            ranking_generator.manual_seed(seed * (2 ** 32) + epoch)
        if negatives is not None:
            extra_rows = negatives.sample(order.cpu().numpy(), batch, seed, epoch)
            extra_rows = torch.as_tensor(extra_rows, device=DEVICE)
        for ids in order.split(batch):
            opt.zero_grad(set_to_none=True)
            emb = model(epoch_x[ids])
            paired = images[ids]
            extra = None if negatives is None else negative_images[extra_rows[ids]]
            loss = contrastive_loss(emb, paired, extra, logit_scale=model.logit_scale,
                                    cosine_weight=cosine_weight, soft_clip=soft_clip)
            if top5_ranking:
                gallery = torch.randperm(len(bank.centers), generator=ranking_generator,
                                        device=DEVICE)[:TOP5_GALLERY_SIZE]
                loss = loss + TOP5_RANK_WEIGHT * bank.top5_loss(emb, paired, ids, gallery)
            elif bank is not None:
                loss = 0.75 * loss + 0.25 * bank.loss(emb, paired, ids)
            loss.backward()
            opt.step()
            losses += loss.detach() * len(ids)
        epoch_loss = float(losses / len(x))
        if not np.isfinite(epoch_loss):
            raise ValueError(f"Non-finite loss: {folder}, epoch {epoch}")
        if ema is not None and epoch >= EMA_FIRST_EPOCH:
            ema.update_parameters(model)
        row = {"epoch": epoch, "loss": epoch_loss}
        if model.logit_scale is not None:
            row["temperature"] = float(contrastive_temperature(model.logit_scale.detach()))
        if val is not None:
            row["val_top1"], row["val_top5"] = accuracy(score(model, val, centers, batch), labels)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        write_log(log_path, rows)
        finished_smoke = stage == "smoke" and row["val_top1"] >= 0.9
        if epoch == 1 or epoch % 10 == 0 or finished_smoke:
            print(row, flush=True)
        if epoch in snapshot_epochs:
            save_weights(folder / f"epoch-{epoch}.pt", model.state_dict())
        if epoch % 5 == 0 and epoch < epochs and not finished_smoke:
            state = dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                         shuffle=generator.get_state(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None)
            if ema is not None:
                state["ema"] = ema.state_dict()
            save_weights(resume, state)
            del state
        if finished_smoke:
            break
    if stage == "smoke" and float(rows[-1]["val_top1"]) < 0.9:
        raise RuntimeError("Small-set memorization below 90%")
    metadata = dict(mean=mean.cpu(), std=std.cpu(), settings=settings, trained_epochs=len(rows))
    save_weights(folder / "checkpoint.pt", dict(model=model.state_dict(), **metadata))
    if ema is not None:
        val_accuracy = accuracy(score(ema.module, val, centers, batch), labels) if val is not None else None
        save_weights(folder / "ema.pt", dict(model=ema.module.state_dict(), validation_accuracy=val_accuracy,
                                             ema_updates=int(ema.n_averaged), **metadata))
    stats = dict(settings, parameters=sum(p.numel() for p in model.parameters()),
                 train_seconds=sum(float(r["seconds"]) for r in rows), trained_epochs=len(rows),
                 peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20 if DEVICE.type == "cuda" else 0)
    if model.logit_scale is not None:
        stats["final_temperature"] = rows[-1]["temperature"]
    if stage == "smoke":
        np.save(folder / "before_reload.npy", score(model, val, centers, batch).cpu().numpy())
    save_json(folder / "training.json", stats)
    resume.unlink(missing_ok=True)


def reload_and_score(config, subject, seed, design, stage="runs"):
    folder = run_dir(stage, config, subject, seed)
    ckpt = torch.load(folder / "checkpoint.pt", map_location=DEVICE)
    model = model_for(config)
    model.load_state_dict(ckpt["model"], strict=True)
    batch = ckpt["settings"]["batch_size"]
    if stage == "smoke":
        _, _, values, centers, labels, _, _ = get_arrays(subject, design, stage)
    else:
        values = (tensor(load_subject(subject, "test")) - ckpt["mean"]) / ckpt["std"]
        centers = tensor(design["test_centers"])
        labels = torch.as_tensor(np.array(design["test_labels"]), device=DEVICE, dtype=torch.long)
    scores = score(model, values, centers, batch)
    arr = scores.cpu().numpy()
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite evaluation scores")
    if stage == "smoke":
        prev = np.load(folder / "before_reload.npy")
        np.testing.assert_array_equal(np.argsort(-prev, axis=1), np.argsort(-arr, axis=1))
    else:
        if arr.shape != (200, 200) or not np.array_equal(labels.cpu().numpy(), np.arange(200)):
            raise ValueError("Unexpected test matrix or class mapping")
    tmp = folder / "scores.tmp.npy"
    np.save(tmp, arr)
    tmp.replace(folder / "scores.npy")
    top1, top5 = accuracy(scores, labels)
    result = dict(load_json(folder / "training.json"), top1=top1, top5=top5)
    save_json(folder / "metrics.json", result)
    return result


def select_epochs():
    histories = [read_log(run_dir("dev", c, s, 0) / "train.csv")
                 for c in CONFIGS for s in (1, 2)]
    if any(len(h) != 100 for h in histories):
        raise ValueError("Eight complete development curves are required")
    return 1 + max(range(100), key=lambda i: (
        sum(float(h[i]["val_top1"]) for h in histories),
        sum(float(h[i]["val_top5"]) for h in histories), -i))


def choose_batch():
    for batch in (128, 64):
        try:
            for config in CONFIGS:
                model = model_for(config)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                x = torch.zeros((batch, 17, 80), device=DEVICE)
                y = torch.randn(batch, 768, device=DEVICE)
                loss = contrastive_loss(model(x), y)
                loss.backward()
                opt.step()
                del model, opt, x, y, loss
            return batch
        except torch.cuda.OutOfMemoryError:
            if batch == 64:
                raise
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "verify-smoke", "develop", "formal", "evaluate"])
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--subject", type=int, choices=range(1, 11))
    parser.add_argument("--seed", type=int, choices=range(5))
    args = parser.parse_args()
    design = load_design()
    OUT.mkdir(exist_ok=True)
    batch_file = OUT / "batch.json"
    if not batch_file.exists():
        save_json(batch_file, {"batch_size": choose_batch()})
    batch = load_json(batch_file)["batch_size"]
    if args.action == "smoke":
        train("baseline", 1, 0, "smoke", 150, 32, design)
    elif args.action == "verify-smoke":
        print(reload_and_score("baseline", 1, 0, design, "smoke"))
    elif args.action == "develop":
        if not (run_dir("smoke", "baseline", 1, 0) / "metrics.json").exists():
            parser.error("Complete smoke and verify-smoke first")
        for config in CONFIGS:
            for subject in (1, 2):
                train(config, subject, 0, "dev", 100, batch, design)
        save_json(OUT / "frozen.json", dict(epochs=select_epochs(), batch_size=batch, configs=CONFIGS,
                  seeds=list(range(5)), learning_rate=LR, temperature=0.07, primary="ab"))
    else:
        frozen = load_json(OUT / "frozen.json")
        for config in ([args.config] if args.config else CONFIGS):
            for subject in ([args.subject] if args.subject else range(1, 11)):
                for seed in ([args.seed] if args.seed is not None else range(5)):
                    if args.action == "formal":
                        train(config, subject, seed, "runs", frozen["epochs"], frozen["batch_size"], design)
                    else:
                        print(reload_and_score(config, subject, seed, design), flush=True)
        if args.action == "evaluate" and args.config is None and args.subject is None and args.seed is None:
            from eeg_evaluate import main as summarize_saved_scores
            summarize_saved_scores(["--runs", str(OUT / "runs"), "--out", str(OUT / "evaluation")])


if __name__ == "__main__":
    main()
