"""Round 08 development experiments for shared EEG pretraining and target adaptation."""
import argparse
import time

import numpy as np
import torch

from eeg_cross_subject import SUBJECTS
from eeg_csls_replication import ADJUSTED, CANDIDATE, K, PAIRS, scoring_arms, training_stats
from eeg_data import load_subject
from eeg_gemini_iterations import BASE, BATCH, PILOT, design_for, freeze, pairs_as_lists, read_development, result_row
from eeg_model import COSINE_REGRESSION_WEIGHT, PAIR_TEMPERATURE, TRANSFER_CONFIG, EEGEncoder, contrastive_loss
from eeg_pooling_dev import evaluate_checkpoint
from eeg_repeat_views import TRIAL_SAMPLING, sample_subset_epoch
from eeg_train import (DEVICE, LR, OUT, load_json, run_dir, save_json, save_weights, seed_run, source_arrays,
                       tensor, train, write_log)
from eeg_trial_augmentation import (CANDIDATE_ARM, OUTPUT, checked_trials, gains, gate, stored_csls, summarize,
                                    top5_wins)


PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
FINETUNE_LR = 0.00005
ARM = f"{TRANSFER_CONFIG}_trialaug"


def source_batches(n_images, n_subjects, batch, seed, epoch):
    rng = np.random.default_rng([seed, epoch, 808])
    orders = [rng.permutation(n_images) for _ in range(n_subjects)]
    n_batches = (n_images + batch - 1) // batch
    for job in rng.permutation(n_subjects * n_batches):
        subject, index = divmod(int(job), n_batches)
        yield subject, orders[subject][index * batch:(index + 1) * batch]


def target_state(state, index):
    target = {name: value for name, value in state.items() if not name.startswith("electrode_adapters.")}
    target["electrode_adapters.0.weight"] = state[f"electrode_adapters.{index}.weight"]
    return target


def replication_gate(rows):
    if sorted((row["subject"], row["seed"]) for row in rows) != list(PAIRS) or any(
            row["queries"] != 1000 for row in rows):
        raise ValueError("Expected all locked replication pairs on the development gallery")
    n = sum(row["queries"] for row in rows)
    return (250 * gains(rows, 5) >= 3 * n and top5_wins(rows) >= 4
            and all(gains([row for row in rows if row["seed"] == seed], 1) >= 0 for seed in (1, 2)))


def pretrain(directory, design, seed):
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.is_file():
        return torch.load(checkpoint, map_location="cpu")
    directory.mkdir(parents=True, exist_ok=True)
    seed_run(seed)
    torch.cuda.reset_peak_memory_stats()
    indices = design["dev_train"]
    trials = [checked_trials(subject, indices, load_subject(subject)) for subject in SUBJECTS]
    full, images, means, stds = source_arrays(SUBJECTS, design)
    epoch_views = torch.empty_like(full)
    model = EEGEncoder(TRANSFER_CONFIG, adapter_count=len(SUBJECTS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    resume = directory / "resume.pt"
    history = []
    peak = 0.0
    if resume.exists():
        state = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        history, peak = state["history"], state["peak_gpu_mib"]
        del state
    visits = len(indices) * len(SUBJECTS)
    for epoch in range(len(history) + 1, PRETRAIN_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        for index, subject in enumerate(SUBJECTS):
            repeats = tensor(trials[index]).sub_(means[index]).div_(stds[index])
            trial_seed = seed * len(SUBJECTS) + subject
            epoch_views[index] = sample_subset_epoch(full[index], repeats, trial_seed, epoch)
            del repeats
        loss_sum = torch.zeros((), device=DEVICE)
        steps = 0
        for index, rows in source_batches(len(indices), len(SUBJECTS), BATCH, seed, epoch):
            ids = torch.as_tensor(rows, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            embedding = model(epoch_views[index, ids], index)
            loss = contrastive_loss(embedding, images[ids], cosine_weight=COSINE_REGRESSION_WEIGHT)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(ids)
            steps += 1
        average = float(loss_sum / visits)
        if not np.isfinite(average):
            raise ValueError(f"Non-finite shared pretraining loss at epoch {epoch}")
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        history.append(dict(epoch=epoch, loss=average, steps=steps, seconds=time.perf_counter() - started))
        print("shared pretraining", history[-1], flush=True)
        if epoch % 5 == 0:
            save_weights(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                      cuda_rng=torch.cuda.get_rng_state(), history=history, peak_gpu_mib=peak))
    write_log(directory / "train.csv", history)
    stats = dict(subjects=list(SUBJECTS), seed=seed, epochs=len(history), images_per_epoch=visits,
                 optimizer_steps=sum(row["steps"] for row in history), parameters=sum(p.numel() for p in model.parameters()),
                 train_seconds=sum(row["seconds"] for row in history), peak_gpu_mib=peak)
    save_weights(checkpoint, dict(model=model.state_dict(), stats=stats))
    resume.unlink(missing_ok=True)
    return torch.load(checkpoint, map_location="cpu")


def main(pretrain_seed=0, *, replication=False):
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA device unavailable")
    round_directory = BASE / "round-08"
    if replication:
        pilot = round_directory / "pilot-seed-1" / "summary.json"
        if not load_json(pilot)["gate_pass"]:
            raise RuntimeError("The pretraining seed 1 pilot did not pass")
        directory = round_directory / "replication"
        split_seed, pairs = 2027, PAIRS
        pretrain_dirs = {seed: directory / "pretrain" / f"seed-{seed}" for seed in (1, 2)}
        seed_settings = dict(pretrain_seeds=[1, 2], pretrain_seed_matches_finetune_seed=True)
        decision = "Top-5 >= +1.2pp and Top-1 >= 0pp for each seed; at least 4/6 subjects improve Top-5"
        control_directory = OUTPUT / "replication"
    else:
        directory = round_directory / ("pilot" if pretrain_seed == 0 else f"pilot-seed-{pretrain_seed}")
        split_seed, pairs = 2026, PILOT
        pretrain_dirs = {pretrain_seed: directory / "pretrain"}
        seed_settings = dict(pretrain_seed=pretrain_seed)
        decision = "Top-5 >= +1pp and Top-1 >= 0pp; at least 3/4 subjects improve Top-5"
        control_directory = OUTPUT / "pilot"
    design = design_for(split_seed)
    labels, templates, indices = (design[key] for key in ("dev_labels", "dev_centers", "dev_train"))
    stage = directory.relative_to(OUT).as_posix()
    control_stage = control_directory.relative_to(OUT).as_posix()
    freeze(directory / "protocol.json", dict(
        candidate=ARM, control=CANDIDATE_ARM, control_stage=control_stage, split_seed=split_seed,
        pairs=pairs_as_lists(pairs), training_classes=1454, training_images=len(indices),
        pretrain_subjects=list(SUBJECTS), pretrain_epochs=PRETRAIN_EPOCHS,
        pretrain_sampling="one visit per image per subject per epoch; shuffle subject-homogeneous batches",
        pretrain_lr=LR, finetune_epochs=FINETUNE_EPOCHS, finetune_lr=FINETUNE_LR,
        optimizer="Adam; restart at fine-tuning", target_parameters=889642, batch_size=BATCH,
        adapter="one bias-free identity-initialized 17x17 matrix per subject; keep only the target adapter",
        temperature=PAIR_TEMPERATURE, cosine_weight=COSINE_REGRESSION_WEIGHT, training_eeg=TRIAL_SAMPLING,
        csls_k=K, calibration="standard four-repeat training EEG from the target only",
        query="1000 matched development queries and 200 fixed templates; standard four-repeat EEG",
        gate=decision, official_test_used=False, **seed_settings))
    controls = {}
    for subject, seed in pairs:
        folder = run_dir(control_stage, CANDIDATE, subject, seed)
        read_development(folder / "validation.npz", design)
        stats = training_stats(folder, CANDIDATE, subject, seed)
        if stats.get("training_eeg") != TRIAL_SAMPLING or stats["parameters"] != 889353:
            raise ValueError(f"Unexpected round 06 trial augmentation control: {folder}")
        controls[subject, seed] = stats, stored_csls(folder / "csls.npz", labels)

    shared = {}
    for seed, pretrain_dir in pretrain_dirs.items():
        shared[seed] = pretrain(pretrain_dir, design, seed)
        save_json(pretrain_dir / "summary.json", shared[seed]["stats"])
    rows = []
    centers = tensor(templates)
    for subject, seed in pairs:
        warm_seed = seed if replication else pretrain_seed
        folder = run_dir(stage, TRANSFER_CONFIG, subject, seed)
        start = pretrain_dirs[warm_seed] / f"target-{subject:02d}.pt"
        save_weights(start, dict(model=target_state(shared[warm_seed]["model"], SUBJECTS.index(subject))))
        full = load_subject(subject)
        if not (folder / "training.json").exists():
            repeats = checked_trials(subject, indices, full)
            train(TRANSFER_CONFIG, subject, seed, stage, FINETUNE_EPOCHS, BATCH, design, split="dev",
                  trial_repeats=repeats, initial_checkpoint=start, lr=FINETUNE_LR)
            del repeats
        validation = folder / "validation.npz"
        _, mean, std = evaluate_checkpoint(TRANSFER_CONFIG, subject, seed, design, tensor(full[design["dev_query"]]),
                                           centers, validation, stage, epochs=FINETUNE_EPOCHS)
        control_path = run_dir(control_stage, CANDIDATE, subject, seed) / "checkpoint.pt"
        control = torch.load(control_path, map_location=DEVICE)
        torch.testing.assert_close(mean, control["mean"], rtol=0, atol=0)
        torch.testing.assert_close(std, control["std"], rtol=0, atol=0)
        read_development(validation, design)
        with np.load(validation) as output:
            ranks = scoring_arms(folder, output["scores"], labels, templates, full[indices],
                                 model_config=TRANSFER_CONFIG)[ADJUSTED]
        stats = training_stats(folder, TRANSFER_CONFIG, subject, seed, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR)
        if (stats.get("training_eeg") != TRIAL_SAMPLING or stats["parameters"] != 889642
                or stats.get("initial_checkpoint") != start.relative_to(OUT).as_posix()):
            raise ValueError(f"Unexpected transfer settings: {folder}")
        control_stats, reference = controls[subject, seed]
        rows.append(result_row(subject, seed, ARM, CANDIDATE_ARM, ranks, reference,
                               {ARM: stats, CANDIDATE_ARM: control_stats}))
    pretrain_seconds = sum(state["stats"]["train_seconds"] for state in shared.values())
    passed = replication_gate(rows) if replication else gate("pilot", rows)
    summarize(directory, rows, gate_pass=passed, official_test_used=False, shared_pretrain_seconds=pretrain_seconds,
              total_training_seconds=pretrain_seconds + sum(row["train_seconds"] for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pretrain-seed", type=int, choices=(0, 1), default=0)
    mode.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    main(args.pretrain_seed, replication=args.replication)
