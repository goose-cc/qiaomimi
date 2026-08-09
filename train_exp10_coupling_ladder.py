#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp10 parameter-coupling ladder diagnostic.

新增两个二元耦合模式：
  a1m       : a1, m vary; gamma fixed
  a1gamma   : a1, gamma vary; m fixed

原有模式仍保留：
  mg
  a1mg_strong
  a1mg_full
  all5_strong
  all5_full

目的
----
在已经确认：
  (m, gamma) 联合变化可以学；
  (a1, m, gamma) 联合变化明显退化；
之后，继续拆分判断到底是：
  a1 <-> m
还是
  a1 <-> gamma
导致主要耦合。

训练原则
--------
- 继续使用实验七的 ParametricInverseTransformer1D。
- 输入仍是 clean 100-point g(q^2)。
- 不加 peak loss / curve loss / gate。
- 只对当前模式中“自由变化”的参数计算归一化 MSE。
- 固定参数不参与 loss。
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from TransformerInverse import ParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalize_parameters
from mc_physics import valid_parameter_mask_numpy
from train_mc_parameter_pool_transformer_loss import (
    amp_autocast,
    atomic_json_save,
    atomic_torch_save,
    choose_device,
    make_grad_scaler,
    set_random_seed,
)

PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")

LOWER = np.array(
    [0.0, 0.0, -0.05, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)],
    dtype=np.float64,
)
UPPER = np.array([0.2, 0.05, 0.05, 2.0, 1.0], dtype=np.float64)

# 固定值与之前 Exp10 保持一致：
# a1=0.10, a2=0.025, a3=0, m=0.8, gamma=0.5
MODE_SPECS = {
    "mg": {
        "free": (3, 4),
        "fixed": {0: 0.10, 1: 0.025, 2: 0.0},
        "a1_min": None,
        "description": "only m,gamma vary; a1=0.10, a2=0.025, a3=0",
    },

    # ---------- 新增：a1 + m ----------
    "a1m": {
        "free": (0, 3),
        "fixed": {1: 0.025, 2: 0.0, 4: 0.5},
        "a1_min": 0.05,
        "description": (
            "a1,m vary; a1 in [0.05,0.2]; "
            "gamma=0.5, a2=0.025, a3=0"
        ),
    },

    # ---------- 新增：a1 + gamma ----------
    "a1gamma": {
        "free": (0, 4),
        "fixed": {1: 0.025, 2: 0.0, 3: 0.8},
        "a1_min": 0.05,
        "description": (
            "a1,gamma vary; a1 in [0.05,0.2]; "
            "m=0.8, a2=0.025, a3=0"
        ),
    },

    "a1mg_strong": {
        "free": (0, 3, 4),
        "fixed": {1: 0.025, 2: 0.0},
        "a1_min": 0.05,
        "description": "a1,m,gamma vary; a1 in [0.05,0.2]; background fixed",
    },
    "a1mg_full": {
        "free": (0, 3, 4),
        "fixed": {1: 0.025, 2: 0.0},
        "a1_min": 0.0,
        "description": "a1,m,gamma vary over full original range; background fixed",
    },
    "all5_strong": {
        "free": (0, 1, 2, 3, 4),
        "fixed": {},
        "a1_min": 0.05,
        "description": "all five vary; a1 in [0.05,0.2]",
    },
    "all5_full": {
        "free": (0, 1, 2, 3, 4),
        "fixed": {},
        "a1_min": 0.0,
        "description": "all five vary over original Exp7 ranges",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Exp10 parameter-coupling ladder")
    p.add_argument("--mode", choices=tuple(MODE_SPECS), required=True)
    p.add_argument("--checkpoint-dir", required=True)

    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--noise-level", type=float, default=0.0)
    p.add_argument("--data-scale", type=float, default=160000.0)
    p.add_argument("--shift", type=float, default=400.0)
    p.add_argument("--s-min", type=float, default=0.1764)
    p.add_argument("--s-max", type=float, default=6.0)
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
    p.add_argument("--physics-dtype", choices=("float32", "float64"), default="float32")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--checkpoint-every-steps", type=int, default=2000)
    p.add_argument("--best-window", type=int, default=100)

    p.add_argument("--val-samples", type=int, default=10000)
    p.add_argument("--val-batch-size", type=int, default=256)

    p.add_argument("--transformer-d-model", type=int, default=64)
    p.add_argument("--transformer-nhead", type=int, default=4)
    p.add_argument("--transformer-num-layers", type=int, default=3)
    p.add_argument("--transformer-dim-feedforward", type=int, default=128)
    p.add_argument("--transformer-dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def make_model(args, device):
    return ParametricInverseTransformer1D(
        input_length=args.input_points,
        output_length=args.output_points,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_encoder_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout,
        y_min=args.q2_min,
        y_max=args.q2_max,
        x_min=args.s_min,
        x_max=args.s_max,
        shift=args.shift,
        data_scale=args.data_scale,
    ).to(device)


def _fill_candidates(rng: np.random.Generator, mode: str, n: int) -> np.ndarray:
    spec = MODE_SPECS[mode]
    p = np.empty((n, 5), dtype=np.float64)

    a1_low = (
        float(spec["a1_min"])
        if spec["a1_min"] is not None
        else LOWER[0]
    )

    p[:, 0] = rng.uniform(a1_low, UPPER[0], n)
    p[:, 1] = rng.uniform(LOWER[1], UPPER[1], n)
    p[:, 2] = rng.uniform(LOWER[2], UPPER[2], n)
    p[:, 3] = rng.uniform(LOWER[3], UPPER[3], n)
    p[:, 4] = rng.uniform(LOWER[4], UPPER[4], n)

    # 固定参数覆盖随机值。
    for idx, value in spec["fixed"].items():
        p[:, int(idx)] = float(value)

    return p


def sample_valid_parameters(
    rng: np.random.Generator,
    mode: str,
    n: int,
) -> np.ndarray:
    accepted = []
    total = 0

    while total < n:
        need = n - total
        candidate_n = max(need * 2, 1024)

        candidates = _fill_candidates(rng, mode, candidate_n)
        valid = valid_parameter_mask_numpy(candidates)
        good = candidates[valid]

        if len(good):
            take = min(need, len(good))
            accepted.append(good[:take])
            total += take

    return np.concatenate(accepted, axis=0).astype(np.float32, copy=False)


def free_loss(pred, target, lower, upper, free_idx):
    pred_n = normalize_parameters(pred, lower, upper)
    true_n = normalize_parameters(target, lower, upper)

    return torch.mean(
        (pred_n[:, free_idx] - true_n[:, free_idx]).square()
    )


def apply_fixed_truth(
    pred: torch.Tensor,
    truth: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    f/g 验证时，固定参数使用真实固定值。
    避免网络没有训练固定参数，却让其误差污染 f/g 指标。
    """
    result = pred.clone()
    free = set(MODE_SPECS[mode]["free"])

    for idx in range(5):
        if idx not in free:
            result[:, idx] = truth[:, idx]

    return result


@torch.no_grad()
def evaluate(model, physics, args, mode, seed):
    model.eval()

    rng = np.random.default_rng(seed)
    params_np = sample_valid_parameters(rng, mode, args.val_samples)

    free_idx = list(MODE_SPECS[mode]["free"])

    sums_abs = np.zeros(5, dtype=np.float64)
    sum_f = 0.0
    sum_g = 0.0
    count = 0

    for start in range(0, len(params_np), args.val_batch_size):
        stop = min(start + args.val_batch_size, len(params_np))

        truth = torch.from_numpy(params_np[start:stop]).to(
            device=physics.device,
            dtype=torch.float32,
        )

        f_true, g_true = physics.make_clean_batch(truth)

        pred = model.predict_parameters(g_true.unsqueeze(1))
        pred_phys = apply_fixed_truth(pred, truth, mode)

        pred_n = normalize_parameters(
            pred,
            model.parameter_lower,
            model.parameter_upper,
        )
        true_n = normalize_parameters(
            truth,
            model.parameter_lower,
            model.parameter_upper,
        )

        abs_n = torch.abs(pred_n - true_n)
        sums_abs += abs_n.sum(dim=0).cpu().numpy()

        f_pred, _, _ = physics.components(pred_phys)
        g_pred = physics.forward_from_parameters(pred_phys)

        f_rel = (
            torch.linalg.vector_norm(f_pred - f_true, dim=1)
            / torch.linalg.vector_norm(f_true, dim=1).clamp_min(1e-12)
        )
        g_rel = (
            torch.linalg.vector_norm(g_pred - g_true, dim=1)
            / torch.linalg.vector_norm(g_true, dim=1).clamp_min(1e-12)
        )

        sum_f += float(f_rel.sum().cpu())
        sum_g += float(g_rel.sum().cpu())
        count += len(truth)

    mae = sums_abs / max(count, 1)
    free_mean = float(np.mean(mae[free_idx]))

    return {
        "mode": mode,
        "description": MODE_SPECS[mode]["description"],
        "samples": int(count),
        "free_parameters": [
            PARAMETER_NAMES[i] for i in free_idx
        ],
        "normalized_abs_mean": {
            name: float(mae[i])
            for i, name in enumerate(PARAMETER_NAMES)
        },
        "free_parameter_mae_mean": free_mean,
        "f_relative_l2_mean": float(sum_f / max(count, 1)),
        "g_relative_l2_mean": float(sum_g / max(count, 1)),
    }


def load_state_dict_file(path: Path):
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location="cpu",
        )

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]

    return obj


def main():
    args = parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")

    # 这个诊断阶段必须保持 0% noise，
    # 否则会把参数耦合和噪声影响混在一起。
    if args.noise_level != 0.0:
        raise ValueError(
            "Exp10 coupling diagnostic should stay at 0% noise first"
        )

    spec = MODE_SPECS[args.mode]
    free_idx = list(spec["free"])

    set_random_seed(args.seed, args.deterministic)

    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    best_path = checkpoint_dir / "best_model.pth"

    model = make_model(args, device)
    physics = OnlinePhysics(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = make_grad_scaler(use_amp)
    rng = np.random.default_rng(args.seed + 17)

    recent = deque(maxlen=args.best_window)
    best_score = float("inf")
    total_samples = 0
    started = time.time()

    print("=" * 80)
    print("Exp10 coupling ladder")
    print(f"mode              : {args.mode}")
    print(f"description       : {spec['description']}")
    print(
        f"free parameters   : "
        f"{[PARAMETER_NAMES[i] for i in free_idx]}"
    )
    print("training noise    : 0%")
    print("loss              : normalized MSE on FREE parameters only")
    print("curve/peak loss   : NONE")
    print(
        f"model parameters  : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print("=" * 80)

    model.train()

    for step in range(1, args.max_steps + 1):
        params_np = sample_valid_parameters(
            rng,
            args.mode,
            args.batch_size,
        )

        truth = torch.from_numpy(params_np).to(
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            _, g_clean = physics.make_clean_batch(truth)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(use_amp):
            pred = model.predict_parameters(g_clean.unsqueeze(1))
            loss = free_loss(
                pred,
                truth,
                model.parameter_lower,
                model.parameter_upper,
                free_idx,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        scaler.step(optimizer)
        scaler.update()

        total_samples += args.batch_size
        recent.append(float(loss.detach().cpu()))

        if step % args.log_every_steps == 0:
            with torch.no_grad():
                pred_n = normalize_parameters(
                    pred,
                    model.parameter_lower,
                    model.parameter_upper,
                )
                true_n = normalize_parameters(
                    truth,
                    model.parameter_lower,
                    model.parameter_upper,
                )

                mae = torch.mean(
                    torch.abs(pred_n - true_n),
                    dim=0,
                ).cpu().numpy()

            rolling = float(np.mean(recent))

            free_text = " ".join(
                f"{PARAMETER_NAMES[i]}={mae[i]:.4f}"
                for i in free_idx
            )

            print(
                f"step={step:6d} "
                f"loss={float(loss):.6e} "
                f"rolling={rolling:.6e} | "
                f"{free_text}"
            )

            # 保持之前 Exp10 的 best 逻辑不变，
            # 方便和已经跑过的 mg / a1mg_strong 严格比较。
            if (
                len(recent) == recent.maxlen
                and rolling < best_score
            ):
                best_score = rolling
                atomic_torch_save(
                    model.state_dict(),
                    best_path,
                )

        if step % args.checkpoint_every_steps == 0:
            atomic_torch_save(
                {
                    "mode": args.mode,
                    "global_step": step,
                    "total_samples": total_samples,
                    "best_score": best_score,
                    "args": vars(args),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                latest_path,
            )

    atomic_torch_save(
        {
            "mode": args.mode,
            "global_step": args.max_steps,
            "total_samples": total_samples,
            "best_score": best_score,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        latest_path,
    )

    # Latest validation
    latest_metrics = evaluate(
        model,
        physics,
        args,
        args.mode,
        seed=args.seed + 100_003,
    )
    latest_metrics["weights"] = "latest"

    atomic_json_save(
        latest_metrics,
        checkpoint_dir / "validation_latest.json",
    )

    # Best validation
    if best_path.exists():
        best_model = make_model(args, device)
        best_model.load_state_dict(
            load_state_dict_file(best_path)
        )

        best_metrics = evaluate(
            best_model,
            physics,
            args,
            args.mode,
            seed=args.seed + 100_003,
        )
        best_metrics["weights"] = "best"

        atomic_json_save(
            best_metrics,
            checkpoint_dir / "validation_best.json",
        )
    else:
        best_metrics = latest_metrics

    summary = {
        "mode": args.mode,
        "description": spec["description"],
        "free_parameters": [
            PARAMETER_NAMES[i] for i in free_idx
        ],
        "max_steps": args.max_steps,
        "total_samples": total_samples,
        "elapsed_seconds": time.time() - started,
        "best_training_score": best_score,
        "latest": latest_metrics,
        "best": best_metrics,
    }

    atomic_json_save(
        summary,
        checkpoint_dir / "exp10_summary.json",
    )

    print("\n" + "=" * 80)
    print("Exp10 finished")

    for label, metrics in (
        ("BEST", best_metrics),
        ("LATEST", latest_metrics),
    ):
        m = metrics["normalized_abs_mean"]

        print(
            f"{label:6s}: "
            f"free_mean={metrics['free_parameter_mae_mean']:.5f} | "
            f"a1={m['a1']:.5f} "
            f"a2={m['a2']:.5f} "
            f"a3={m['a3']:.5f} "
            f"m={m['m']:.5f} "
            f"gamma={m['gamma']:.5f} | "
            f"f={metrics['f_relative_l2_mean']:.5f} "
            f"g={metrics['g_relative_l2_mean']:.5f}"
        )

    print(f"output: {checkpoint_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
