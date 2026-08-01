"""
This file is an integration example, not a replacement for your model definition.

Copy the dataset/DataLoader part into the existing main.py. The online dataset
already yields complete batches, so DataLoader must use batch_size=None.
"""

from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from online_noise_dataset import OnlineTruthBatchDataset


def build_online_loader(args: argparse.Namespace, epoch: int) -> tuple:
    dataset = OnlineTruthBatchDataset(
        root=args.truth_shards,
        batch_size=args.batch_size,
        noise_level=0.09,
        noises_per_truth=1000,
        master_seed=args.master_seed,
        data_scale=args.data_scale,
        shuffle=True,
        epoch=epoch,
        # For a smoke test only:
        max_shards=args.max_shards,
        max_noise_ids=args.max_noise_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,      # Important: dataset already returns a batch.
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return dataset, loader


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    running_loss = 0.0
    batch_count = 0

    for g_noisy, u_truth, truth_id, noise_id in loader:
        g_noisy = g_noisy.to(device, non_blocking=True)
        u_truth = u_truth.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(g_noisy)

        # Adapt only if your model returns [B, 1, L].
        if prediction.ndim == 3 and prediction.shape[1] == 1:
            prediction = prediction[:, 0, :]

        loss = criterion(prediction, u_truth)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        batch_count += 1

    return running_loss / max(batch_count, 1)


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--truth-shards", default="./truth_shards")
    parser.add_argument("--master-seed", type=int, default=20260801)
    parser.add_argument("--data-scale", type=float, default=160000.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    # Only for smoke tests. Keep both as None for the complete logical epoch.
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-noise-ids", type=int, default=None)


# Existing main.py should use this pattern:
#
# for epoch in range(num_epochs):
#     dataset, loader = build_online_loader(args, epoch)
#     dataset.set_epoch(epoch)
#     loss = train_one_epoch(model, optimizer, criterion, loader, device)
#     print(f"Epoch {epoch+1}: loss={loss:.6g}")
#
# With max_shards=None and max_noise_ids=None, one epoch traverses every one of
# the 1000 deterministic noise IDs for every stored truth.
