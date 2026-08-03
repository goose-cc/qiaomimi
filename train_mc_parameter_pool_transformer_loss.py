#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1.6亿蒙特卡洛参数池的分批、限时训练入口。

严格规则：
1. 只使用原项目中的三个模型类：
   - PINet.PeakInversionCNN
   - UNetLike.UNetLikeModel
   - TransformerInverse.InverseTransformer1D
2. 本文件不定义任何 BuiltinTransformer / BuiltinCNN / BuiltinUNet。
3. 原项目模型导入失败时立即停止，不做静默回退。
4. 每次启动都会打印模型类、模块、源码文件和参数量。
5. 从 parameters.dat 顺序读取五维参数：
   [a1, a2, a3, m, gamma]
6. 在线计算 rho(s)、u(s)、稳定正向积分和 9% RMS 高斯白噪声。
7. 使用监督项 + 梯度项 + 物理积分一致性项的可配置组合 loss。
8. 支持按时间停止、检查点续训和参数池循环覆盖。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.legendre import leggauss

from mc_inverse_loss import MonteCarloInverseLoss
from mc_pool_config import DEFAULT_PHYSICS


SCRIPT_VERSION = 4
PARAMETER_COLUMNS = 5
PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")

# 新物理问题的默认范围
DEFAULT_S_MIN = DEFAULT_PHYSICS.s_min
DEFAULT_S_MAX = DEFAULT_PHYSICS.s_max
DEFAULT_Q2_MIN = DEFAULT_PHYSICS.q2_min
DEFAULT_Q2_MAX = DEFAULT_PHYSICS.q2_max
DEFAULT_SHIFT = DEFAULT_PHYSICS.shift
DEFAULT_DATA_SCALE = DEFAULT_PHYSICS.data_scale
DEFAULT_NOISE_LEVEL = DEFAULT_PHYSICS.noise_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用原项目模型训练 1.6 亿蒙特卡洛参数池。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--pool-dir", required=True, help="含 parameters.dat 的参数池目录")
    parser.add_argument("--checkpoint-dir", required=True, help="本实验检查点目录")
    parser.add_argument(
        "--model-type",
        choices=("transformer", "cnn", "unet"),
        default="transformer",
        help="只允许选择原项目已有模型",
    )

    parser.add_argument("--active-block-size", type=int, default=200_000)
    parser.add_argument("--precompute-chunk-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--integration-points", type=int, default=512)
    parser.add_argument("--input-points", type=int, default=100)
    parser.add_argument("--output-points", type=int, default=100)

    parser.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL)
    parser.add_argument("--data-scale", type=float, default=DEFAULT_DATA_SCALE)
    parser.add_argument("--s-min", type=float, default=DEFAULT_S_MIN)
    parser.add_argument("--s-max", type=float, default=DEFAULT_S_MAX)
    parser.add_argument("--q2-min", type=float, default=DEFAULT_Q2_MIN)
    parser.add_argument("--q2-max", type=float, default=DEFAULT_Q2_MAX)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    parser.add_argument(
        "--physics-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="在线物理计算精度；GPU 上 float64 会明显变慢",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--loss-profile",
        choices=(
            "base", "mse_grad", "huber_grad", "pinn",
            "pinn_smooth", "pinn_tikhonov", "pinn_huber", "pinn_full",
        ),
        default="pinn",
        help="组合损失类型；推荐先用 pinn",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=("relative", "absolute"),
        default="relative",
        help="relative 会按每条样本的 RMS 归一化，避免大振幅样本支配训练",
    )
    parser.add_argument(
        "--physics-target",
        choices=("clean", "noisy", "discrete-clean"),
        default="clean",
        help=(
            "physics loss 的 g 参考：clean=高精度无噪声正演；"
            "noisy=网络输入；discrete-clean=由100点真值再做梯形积分"
        ),
    )
    parser.add_argument("--lambda-grad", type=float, default=0.1)
    parser.add_argument("--lambda-physics", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-tv", type=float, default=0.0)
    parser.add_argument("--lambda-tikhonov", type=float, default=0.0)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--loss-eps", type=float, default=1e-8)

    # 与原项目 Transformer 默认设置保持一致
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-num-layers", type=int, default=3)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=128)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    # Python 3.8 兼容：BooleanOptionalAction 是 Python 3.9 才加入的。
    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        "--transformer-normalize-coordinates",
        dest="transformer_normalize_coordinates",
        action="store_true",
        help="将 x/q2 位置坐标映射到 [-1,1] 后再嵌入；本问题必须启用",
    )
    coord_group.add_argument(
        "--no-transformer-normalize-coordinates",
        dest="transformer_normalize_coordinates",
        action="store_false",
        help="关闭坐标归一化（本问题不推荐）",
    )
    parser.set_defaults(transformer_normalize_coordinates=True)

    rms_group = parser.add_mutually_exclusive_group()
    rms_group.add_argument(
        "--transformer-rms-normalize-io",
        dest="transformer_rms_normalize_io",
        action="store_true",
        help="每条 g 按自身 RMS 归一化，网络输出后乘回同一尺度",
    )
    rms_group.add_argument(
        "--no-transformer-rms-normalize-io",
        dest="transformer_rms_normalize_io",
        action="store_false",
        help="关闭逐样本 RMS 归一化（本问题不推荐）",
    )
    parser.set_defaults(transformer_rms_normalize_io=True)
    parser.add_argument(
        "--transformer-rms-eps",
        type=float,
        default=1e-8,
        help="样本 RMS 归一化的数值下限",
    )

    parser.add_argument("--max-hours", type=float, default=23.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 表示不限制训练步数，只按 max-hours 停止",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=2000)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--best-window", type=int, default=100)

    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
    )
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="CUDA 上启用混合精度")
    parser.add_argument("--deterministic", action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fresh",
        action="store_true",
        help="删除本检查点目录中的 latest/best 文件，从原始模型重新开始",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="从 latest_checkpoint.pth 继续",
    )

    parser.add_argument(
        "--require-complete-pool",
        action="store_true",
        help="metadata.json 未标记 complete=true 时拒绝训练",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "active_block_size": args.active_block_size,
        "precompute_chunk_size": args.precompute_chunk_size,
        "batch_size": args.batch_size,
        "integration_points": args.integration_points,
        "input_points": args.input_points,
        "output_points": args.output_points,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "log_every_steps": args.log_every_steps,
        "best_window": args.best_window,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0，当前为 {value}")

    if args.max_hours <= 0 and args.max_steps <= 0:
        raise ValueError("--max-hours 和 --max-steps 至少有一个必须大于 0")
    if not (0.0 <= args.noise_level):
        raise ValueError("--noise-level 不能为负数")
    if args.s_max <= args.s_min:
        raise ValueError("--s-max 必须大于 --s-min")
    if args.q2_max <= args.q2_min:
        raise ValueError("--q2-max 必须大于 --q2-min")
    if args.data_scale <= 0:
        raise ValueError("--data-scale 必须大于 0")
    if args.transformer_d_model % args.transformer_nhead != 0:
        raise ValueError("transformer d_model 必须能被 nhead 整除")
    if args.transformer_rms_eps <= 0:
        raise ValueError("--transformer-rms-eps 必须大于 0")
    for name in (
        "lambda_grad", "lambda_physics", "lambda_smooth",
        "lambda_tv", "lambda_tikhonov",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} 不能为负数")
    if args.huber_beta <= 0:
        raise ValueError("--huber-beta 必须大于 0")
    if args.loss_eps <= 0:
        raise ValueError("--loss-eps 必须大于 0")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False。")
    if name == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("请求了 XPU，但当前 PyTorch 没有可用的 torch.xpu。")
    return torch.device(name)


def set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def build_original_project_model(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    只从原项目文件导入模型。任何导入或构造失败都会直接停止。
    """
    expected_class = {
        "transformer": "InverseTransformer1D",
        "cnn": "PeakInversionCNN",
        "unet": "UNetLikeModel",
    }[args.model_type]

    try:
        if args.model_type == "transformer":
            from TransformerInverse import InverseTransformer1D

            model = InverseTransformer1D(
                input_length=args.input_points,
                output_length=args.output_points,
                d_model=args.transformer_d_model,
                nhead=args.transformer_nhead,
                num_encoder_layers=args.transformer_num_layers,
                num_decoder_layers=args.transformer_num_layers,
                dim_feedforward=args.transformer_dim_feedforward,
                dropout=args.transformer_dropout,
                # 新问题中，输出坐标是 s，输入坐标是 q^2
                x_min=args.s_min,
                x_max=args.s_max,
                y_min=args.q2_min,
                y_max=args.q2_max,
                normalize_coordinates=args.transformer_normalize_coordinates,
                rms_normalize_io=args.transformer_rms_normalize_io,
                rms_eps=args.transformer_rms_eps,
            )
        elif args.model_type == "cnn":
            from PINet import PeakInversionCNN

            model = PeakInversionCNN(
                input_length=args.input_points,
                output_length=args.output_points,
            )
        else:
            from UNetLike import UNetLikeModel

            # 原项目 ModelTools.py 也是直接使用 UNetLikeModel()
            model = UNetLikeModel()
    except Exception as exc:
        required_file = {
            "transformer": "TransformerInverse.py",
            "cnn": "PINet.py（以及 ResidualBlock.py）",
            "unet": "UNetLike.py",
        }[args.model_type]
        raise RuntimeError(
            f"无法导入或构造原项目 {args.model_type} 模型。\n"
            f"请确认脚本位于项目根目录，并且存在 {required_file}。\n"
            "为避免误用备用模型，程序已停止；本脚本不会回退到任何内置网络。"
        ) from exc

    if model.__class__.__name__ != expected_class:
        raise RuntimeError(
            f"模型类不符合预期：应为 {expected_class}，实际为 "
            f"{model.__class__.__name__}。训练已停止。"
        )

    source_file = Path(inspect.getfile(model.__class__)).resolve()
    this_file = Path(__file__).resolve()
    if source_file == this_file or model.__class__.__name__.startswith("Builtin"):
        raise RuntimeError(
            "检测到模型来自训练脚本本身或名称为 Builtin*，训练已停止。"
        )

    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    info = {
        "model_type": args.model_type,
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "model_source_file": str(source_file),
        "model_parameter_count": int(parameter_count),
    }

    print("=" * 72)
    print("严格使用原项目模型")
    print(f"实际模型类: {info['model_class']}")
    print(f"模型所属模块: {info['model_module']}")
    print(f"模型源码文件: {info['model_source_file']}")
    print(f"模型参数量: {parameter_count:,}")
    print("=" * 72)
    return model, info


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 JSON 文件：{path}") from exc


def recursive_find_number(data: Any, keys: Tuple[str, ...]) -> Optional[int]:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value_int = int(value)
                    if value_int >= 0:
                        return value_int
        for value in data.values():
            found = recursive_find_number(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = recursive_find_number(value, keys)
            if found is not None:
                return found
    return None


def recursive_find_bool(data: Any, keys: Tuple[str, ...]) -> Optional[bool]:
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], bool):
                return data[key]
        for value in data.values():
            found = recursive_find_bool(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = recursive_find_bool(value, keys)
            if found is not None:
                return found
    return None


def open_parameter_pool(
    pool_dir: Path,
    require_complete: bool,
) -> Tuple[np.memmap, Dict[str, Any], int]:
    data_path = pool_dir / "parameters.dat"
    metadata_path = pool_dir / "metadata.json"

    if not data_path.exists():
        raise FileNotFoundError(f"找不到参数池文件：{data_path}")

    row_bytes = PARAMETER_COLUMNS * np.dtype(np.float32).itemsize
    file_size = data_path.stat().st_size
    if file_size <= 0 or file_size % row_bytes != 0:
        raise RuntimeError(
            f"parameters.dat 大小异常：{file_size} 字节；"
            f"应当是 {row_bytes} 字节的整数倍。"
        )

    file_rows = file_size // row_bytes
    metadata = read_json(metadata_path)

    generated_keys = (
        "generated_count",
        "generated",
        "num_generated",
        "generated_truths",
        "valid_truths",
        "written_rows",
        "completed_truths",
    )
    generated_rows = recursive_find_number(metadata, generated_keys)
    complete = recursive_find_bool(metadata, ("complete", "completed", "is_complete"))

    if require_complete and complete is not True:
        raise RuntimeError(
            "要求完整参数池，但 metadata.json 没有明确标记 complete=true。"
        )

    if generated_rows is None or generated_rows == 0:
        usable_rows = int(file_rows)
    else:
        usable_rows = min(int(file_rows), int(generated_rows))

    if usable_rows <= 0:
        raise RuntimeError("参数池中没有可训练的参数。")

    pool = np.memmap(
        data_path,
        dtype=np.float32,
        mode="r",
        shape=(int(file_rows), PARAMETER_COLUMNS),
    )

    print("=" * 72)
    print("参数池信息")
    print(f"文件: {data_path.resolve()}")
    print(f"文件行数: {file_rows:,}")
    print(f"可用行数: {usable_rows:,}")
    print(f"参数列顺序: {', '.join(PARAMETER_NAMES)}")
    print(f"metadata complete: {complete}")
    print("=" * 72)
    return pool, metadata, usable_rows


class OnlinePhysics:
    """
    在训练设备上批量计算：
        params -> rho(s) -> u_scaled(s) -> g_clean_scaled(q^2)
    """

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.device = device
        self.dtype = torch.float64 if args.physics_dtype == "float64" else torch.float32
        self.model_dtype = torch.float32

        self.noise_level = float(args.noise_level)
        self.data_scale = float(args.data_scale)
        self.shift = float(args.shift)

        self.s_output = torch.linspace(
            args.s_min,
            args.s_max,
            args.output_points,
            dtype=self.dtype,
            device=device,
        )
        self.q2_grid = torch.linspace(
            args.q2_min,
            args.q2_max,
            args.input_points,
            dtype=self.dtype,
            device=device,
        )

        nodes, weights = leggauss(args.integration_points)
        self.gl_nodes = torch.as_tensor(nodes, dtype=self.dtype, device=device)
        self.gl_weights = torch.as_tensor(weights, dtype=self.dtype, device=device)
        self.s_min = float(args.s_min)
        self.s_max = float(args.s_max)

        # 平滑线性背景使用固定 Gauss-Legendre；窄 Lorentz 峰使用 atan 变量代换。
        s_mid = 0.5 * (args.s_max + args.s_min)
        s_half = 0.5 * (args.s_max - args.s_min)
        self.s_fixed = s_mid + s_half * self.gl_nodes
        self.s_fixed_weights = s_half * self.gl_weights
        self.background_kernel = self.s_fixed_weights[:, None] / (
            self.s_fixed[:, None] - self.q2_grid[None, :]
        )
        self.q_block_size = 25

    @staticmethod
    def _rho(s: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        a1 = params[:, 0:1]
        a2 = params[:, 1:2]
        a3 = params[:, 2:3]
        m = params[:, 3:4]
        gamma = params[:, 4:5]

        width = m * gamma
        resonance = (
            (a1 / math.pi)
            * width
            / ((s[None, :] - m) ** 2 + width**2)
        )
        return resonance + a2 * s[None, :] + a3

    def make_clean_batch(
        self,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = params.to(device=self.device, dtype=self.dtype)

        rho_output = self._rho(self.s_output, params)
        u_target_scaled = (
            rho_output
            / (self.s_output[None, :] + self.shift) ** 2
            * self.data_scale
        )

        a1 = params[:, 0:1]
        a2 = params[:, 1:2]
        a3 = params[:, 2:3]
        mass = params[:, 3:4]
        gamma = params[:, 4:5]
        width = mass * gamma

        # 线性背景：(a2*s+a3)/(s+shift)^2
        background = (
            a2 * self.s_fixed[None, :] + a3
        ) / (self.s_fixed[None, :] + self.shift) ** 2
        g_clean = background @ self.background_kernel

        # Lorentz 共振峰采用 z=atan((s-m)/(m*gamma))，避免窄峰被固定 s 网格漏掉。
        z0 = torch.atan((self.s_min - mass) / width)
        z1 = torch.atan((self.s_max - mass) / width)
        z_mid = 0.5 * (z0 + z1)
        z_half = 0.5 * (z1 - z0)
        z = z_mid + z_half * self.gl_nodes[None, :]
        s_res = mass + width * torch.tan(z)
        coefficient = (
            (a1 / math.pi)
            * z_half
            * self.gl_weights[None, :]
            / (s_res + self.shift) ** 2
        )
        for start in range(0, self.q2_grid.numel(), self.q_block_size):
            stop = min(start + self.q_block_size, self.q2_grid.numel())
            denominator = s_res[:, :, None] - self.q2_grid[None, None, start:stop]
            g_clean[:, start:stop] = g_clean[:, start:stop] + torch.sum(
                coefficient[:, :, None] / denominator, dim=1
            )
        g_clean_scaled = self.data_scale * g_clean

        if not torch.isfinite(u_target_scaled).all():
            raise FloatingPointError("u_target_scaled 中出现 NaN 或 Inf。")
        if not torch.isfinite(g_clean_scaled).all():
            raise FloatingPointError("g_clean_scaled 中出现 NaN 或 Inf。")

        # 模型权重通常是 float32
        return (
            u_target_scaled.to(self.model_dtype),
            g_clean_scaled.to(self.model_dtype),
        )

    def add_noise(self, g_clean_scaled: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            torch.mean(g_clean_scaled.square(), dim=1, keepdim=True)
            + 1e-24
        )
        noise = (
            self.noise_level
            * rms
            * torch.randn_like(g_clean_scaled)
        )
        return g_clean_scaled + noise


def normalize_prediction_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(1)
    elif (
        prediction.ndim == 3
        and prediction.shape[1] != 1
        and prediction.shape[2] == 1
    ):
        prediction = prediction.transpose(1, 2)

    if prediction.shape != target.shape:
        raise RuntimeError(
            "模型输出形状与标签不一致："
            f"prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}。"
        )
    return prediction


def make_grad_scaler(use_amp: bool):
    """Create a GradScaler without emitting deprecated torch.cuda.amp warnings."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            try:
                return torch.amp.GradScaler(device="cuda", enabled=use_amp)
            except TypeError:
                return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast(use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json_save(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:
        raise RuntimeError("检查点中的随机状态无法恢复。") from exc


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def remove_fresh_checkpoint_files(checkpoint_dir: Path) -> None:
    names = (
        "latest_checkpoint.pth",
        "best_model.pth",
        "best_model_info.json",
        "training_summary.json",
    )
    for name in names:
        path = checkpoint_dir / name
        if path.exists():
            path.unlink()
            print(f"--fresh 已删除旧文件: {path}")


def build_checkpoint(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    model_info: Dict[str, Any],
    usable_rows: int,
    next_parameter_id: int,
    pool_cycle: int,
    global_step: int,
    total_samples: int,
    total_elapsed_seconds: float,
    best_score: float,
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        **model_info,
        "args": vars(args),
        "pool_usable_rows": int(usable_rows),
        "next_parameter_id": int(next_parameter_id),
        "pool_cycle": int(pool_cycle),
        "global_step": int(global_step),
        "total_samples": int(total_samples),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "best_score": float(best_score),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "rng_state": capture_rng_state(),
    }


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    model_info: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到续训检查点：{path}")

    # 完整训练断点含优化器和随机状态，只加载本项目自己生成、可信的文件。
    # PyTorch 2.6+ 默认 weights_only=True，因此这里必须显式设为 False。
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    saved_class = checkpoint.get("model_class", "")
    saved_source = checkpoint.get("model_source_file", "")

    if saved_class.startswith("Builtin") or "train_mc_parameter_pool_day.py" in str(saved_source):
        raise RuntimeError(
            "这个检查点由旧版内置 Builtin 模型生成，不能加载到原项目模型。\n"
            "请改用一个新的 checkpoint-dir，或使用 --fresh 删除旧检查点后重新训练。"
        )

    if saved_class and saved_class != model_info["model_class"]:
        raise RuntimeError(
            f"检查点模型类为 {saved_class}，当前模型类为 "
            f"{model_info['model_class']}，不能续训。"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state(optimizer, device)

    scaler_state = checkpoint.get("amp_scaler_state_dict")
    if scaler.is_enabled() and scaler_state:
        scaler.load_state_dict(scaler_state)

    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        restore_rng_state(rng_state)

    print(f"已恢复检查点: {path.resolve()}")
    print(
        "续训位置: "
        f"cycle={checkpoint.get('pool_cycle', 0)}, "
        f"next_parameter_id={checkpoint.get('next_parameter_id', 0):,}, "
        f"global_step={checkpoint.get('global_step', 0):,}"
    )
    return checkpoint


def validate_resume_compatibility(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    *,
    pool_dir: Path,
    usable_rows: int,
) -> None:
    """Reject accidental continuation with a different experiment definition."""
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        raise RuntimeError("检查点缺少可验证的 args，不能安全续训。请使用 --fresh。")

    critical_keys = (
        "model_type", "input_points", "output_points",
        "transformer_d_model", "transformer_nhead",
        "transformer_num_layers", "transformer_dim_feedforward",
        "transformer_dropout", "transformer_normalize_coordinates",
        "transformer_rms_normalize_io", "transformer_rms_eps",
        "s_min", "s_max", "q2_min", "q2_max", "shift",
        "data_scale", "noise_level", "integration_points",
        "physics_dtype", "loss_profile", "loss_normalization",
        "physics_target", "lambda_grad", "lambda_physics",
        "lambda_smooth", "lambda_tv", "lambda_tikhonov",
        "huber_beta", "loss_eps", "learning_rate", "weight_decay",
        "grad_clip", "batch_size", "amp",
    )
    mismatches = []
    for key in critical_keys:
        if key not in saved_args:
            continue
        old = saved_args[key]
        new = getattr(args, key)
        if isinstance(old, float) or isinstance(new, float):
            try:
                equal = math.isclose(float(old), float(new), rel_tol=1e-12, abs_tol=1e-12)
            except (TypeError, ValueError):
                equal = old == new
        else:
            equal = old == new
        if not equal:
            mismatches.append(f"{key}: checkpoint={old!r}, current={new!r}")

    saved_pool = saved_args.get("pool_dir")
    if saved_pool:
        try:
            if Path(saved_pool).resolve() != pool_dir.resolve():
                mismatches.append(
                    f"pool_dir: checkpoint={Path(saved_pool).resolve()}, current={pool_dir.resolve()}"
                )
        except OSError:
            pass

    saved_rows = checkpoint.get("pool_usable_rows")
    if saved_rows is not None and int(saved_rows) != int(usable_rows):
        mismatches.append(
            f"pool_usable_rows: checkpoint={int(saved_rows)}, current={int(usable_rows)}"
        )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "续训参数与原检查点不一致：\n  - " + details +
            "\n为避免混合两个实验，请恢复原参数，或换新 checkpoint-dir 并使用 --fresh。"
        )


def should_stop(
    *,
    args: argparse.Namespace,
    start_time: float,
    global_step: int,
    start_global_step: int,
) -> Optional[str]:
    elapsed_hours = (time.monotonic() - start_time) / 3600.0
    if args.max_hours > 0 and elapsed_hours >= args.max_hours:
        return f"达到最大运行时间 {args.max_hours} 小时"

    run_steps = global_step - start_global_step
    if args.max_steps > 0 and run_steps >= args.max_steps:
        return f"达到本次最大步数 {args.max_steps}"
    return None


def save_best_model(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    model_info: Dict[str, Any],
    score: float,
    global_step: int,
    pool_cycle: int,
    next_parameter_id: int,
) -> None:
    # 保存纯 state_dict，便于直接 model.load_state_dict(torch.load(...))
    atomic_torch_save(model.state_dict(), checkpoint_dir / "best_model.pth")
    atomic_json_save(
        {
            **model_info,
            "score_definition": "最近若干训练 batch 的平均组合 loss；不是验证集指标",
            "best_score": float(score),
            "global_step": int(global_step),
            "pool_cycle": int(pool_cycle),
            "next_parameter_id": int(next_parameter_id),
            "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        checkpoint_dir / "best_model_info.json",
    )


def train(args: argparse.Namespace) -> None:
    validate_args(args)

    pool_dir = Path(args.pool_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    device = choose_device(args.device)
    set_random_seed(args.seed, args.deterministic)
    print(f"训练设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    pool, metadata, usable_rows = open_parameter_pool(
        pool_dir,
        args.require_complete_pool,
    )

    model, model_info = build_original_project_model(args, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = MonteCarloInverseLoss(
        s_min=args.s_min,
        s_max=args.s_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        output_points=args.output_points,
        input_points=args.input_points,
        profile=args.loss_profile,
        normalization=args.loss_normalization,
        lambda_grad=args.lambda_grad,
        lambda_physics=args.lambda_physics,
        lambda_smooth=args.lambda_smooth,
        lambda_tv=args.lambda_tv,
        lambda_tikhonov=args.lambda_tikhonov,
        huber_beta=args.huber_beta,
        eps=args.loss_eps,
    ).to(device)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)
    physics = OnlinePhysics(args, device)

    latest_path = checkpoint_dir / "latest_checkpoint.pth"

    if args.fresh:
        remove_fresh_checkpoint_files(checkpoint_dir)
    elif latest_path.exists() and not args.resume:
        raise RuntimeError(
            f"检测到已有检查点：{latest_path}\n"
            "为避免误覆盖，请明确使用 --resume 或 --fresh。"
        )

    next_parameter_id = 0
    pool_cycle = 0
    global_step = 0
    total_samples = 0
    previous_elapsed_seconds = 0.0
    best_score = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_info=model_info,
            device=device,
        )
        validate_resume_compatibility(
            checkpoint, args, pool_dir=pool_dir, usable_rows=usable_rows
        )
        next_parameter_id = int(checkpoint.get("next_parameter_id", 0))
        pool_cycle = int(checkpoint.get("pool_cycle", 0))
        global_step = int(checkpoint.get("global_step", 0))
        total_samples = int(checkpoint.get("total_samples", 0))
        previous_elapsed_seconds = float(
            checkpoint.get("total_elapsed_seconds", 0.0)
        )
        best_score = float(checkpoint.get("best_score", float("inf")))

        if not 0 <= next_parameter_id < usable_rows:
            raise RuntimeError(
                "检查点中的 next_parameter_id 超出当前参数池范围。"
            )

    model.train()
    loss_window = deque(maxlen=args.best_window)
    start_time = time.monotonic()
    start_global_step = global_step
    run_samples = 0
    stop_reason: Optional[str] = None

    print("=" * 72)
    print("开始训练")
    print(f"noise_level: {args.noise_level:.2%} RMS Gaussian")
    print(f"data_scale: {args.data_scale:g}")
    print(f"batch_size: {args.batch_size}")
    print(f"active_block_size: {args.active_block_size:,}")
    print(f"precompute_chunk_size: {args.precompute_chunk_size:,}")
    print(f"integration_points: {args.integration_points}")
    print(f"loss_profile: {args.loss_profile}")
    print(f"loss_normalization: {args.loss_normalization}")
    print(f"physics_target: {args.physics_target}")
    print(
        "transformer scaling: "
        f"coordinate_norm={args.transformer_normalize_coordinates}, "
        f"rms_io_norm={args.transformer_rms_normalize_io}"
    )
    if args.model_type == "transformer" and not args.transformer_normalize_coordinates:
        print(
            "警告：当前 q2 坐标绝对值很大，关闭坐标归一化会让位置嵌入淹没 g 信号。"
        )
    if args.model_type == "transformer" and not args.transformer_rms_normalize_io:
        print(
            "警告：当前 g_scaled 的幅度通常远小于 1，关闭 RMS I/O 归一化容易造成平均曲线塌缩。"
        )
    print(
        "loss weights: "
        f"grad={args.lambda_grad:g}, physics={args.lambda_physics:g}, "
        f"smooth={args.lambda_smooth:g}, tv={args.lambda_tv:g}, "
        f"tikhonov={args.lambda_tikhonov:g}"
    )
    print(f"max_hours: {args.max_hours}")
    print(f"max_steps: {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    print("=" * 72)

    try:
        while stop_reason is None:
            if next_parameter_id >= usable_rows:
                next_parameter_id = 0
                pool_cycle += 1
                print(f"参数池完成一轮，进入 pool_cycle={pool_cycle}")

            block_size = min(
                args.active_block_size,
                usable_rows - next_parameter_id,
            )
            if block_size <= 0:
                next_parameter_id = 0
                pool_cycle += 1
                continue

            # 只复制当前活跃块，避免长期持有 memmap 视图。
            block = np.array(
                pool[next_parameter_id : next_parameter_id + block_size],
                dtype=np.float32,
                copy=True,
            )

            block_offset = 0
            while block_offset < block_size and stop_reason is None:
                stop_reason = should_stop(
                    args=args,
                    start_time=start_time,
                    global_step=global_step,
                    start_global_step=start_global_step,
                )
                if stop_reason is not None:
                    break

                chunk_end = min(
                    block_offset + args.precompute_chunk_size,
                    block_size,
                )
                chunk_np = block[block_offset:chunk_end]
                params = torch.from_numpy(chunk_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=(device.type == "cuda"),
                )

                target_scaled, g_clean_scaled = physics.make_clean_batch(params)
                del params

                local_offset = 0
                chunk_size = chunk_end - block_offset

                while local_offset < chunk_size:
                    stop_reason = should_stop(
                        args=args,
                        start_time=start_time,
                        global_step=global_step,
                        start_global_step=start_global_step,
                    )
                    if stop_reason is not None:
                        break

                    batch_end = min(
                        local_offset + args.batch_size,
                        chunk_size,
                    )
                    current_batch = batch_end - local_offset

                    target = target_scaled[local_offset:batch_end].unsqueeze(1)
                    g_clean = g_clean_scaled[local_offset:batch_end]
                    g_noisy = physics.add_noise(g_clean).unsqueeze(1)

                    optimizer.zero_grad(set_to_none=True)

                    with amp_autocast(use_amp):
                        prediction = model(g_noisy)
                        prediction = normalize_prediction_shape(
                            prediction,
                            target,
                        )
                        if args.physics_target == "clean":
                            g_reference = g_clean.unsqueeze(1)
                        elif args.physics_target == "noisy":
                            g_reference = g_noisy
                        else:
                            # 与100点 f_true 完全离散一致，适合极窄峰标签不可解析时。
                            g_reference = criterion.physics_forward_integral(target)
                        loss, loss_logs = criterion(
                            prediction, target, g_reference
                        )

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"global_step={global_step} 的 loss 非有限：{loss.item()}"
                        )

                    scaler.scale(loss).backward()

                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.grad_clip,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                    loss_value = float(loss.detach().cpu().item())
                    loss_window.append(loss_value)

                    global_step += 1
                    total_samples += current_batch
                    run_samples += current_batch
                    next_parameter_id += current_batch
                    local_offset = batch_end

                    if next_parameter_id >= usable_rows:
                        next_parameter_id = 0
                        pool_cycle += 1

                    rolling_score = float(np.mean(loss_window))
                    if (
                        len(loss_window) == args.best_window
                        and rolling_score < best_score
                    ):
                        best_score = rolling_score
                        save_best_model(
                            checkpoint_dir=checkpoint_dir,
                            model=model,
                            model_info=model_info,
                            score=best_score,
                            global_step=global_step,
                            pool_cycle=pool_cycle,
                            next_parameter_id=next_parameter_id,
                        )

                    if global_step % args.log_every_steps == 0:
                        elapsed = time.monotonic() - start_time
                        throughput = run_samples / max(elapsed, 1e-9)
                        print(
                            f"step={global_step:,} "
                            f"cycle={pool_cycle} "
                            f"next_id={next_parameter_id:,}/{usable_rows:,} "
                            f"loss={loss_value:.6e} "
                            f"data={loss_logs['data_mse']:.3e} "
                            f"grad={loss_logs['grad']:.3e} "
                            f"phys={loss_logs['physics']:.3e} "
                            f"rolling={rolling_score:.6e} "
                            f"throughput={throughput:,.1f} samples/s"
                        )

                    if global_step % args.checkpoint_every_steps == 0:
                        elapsed_total = (
                            previous_elapsed_seconds
                            + time.monotonic()
                            - start_time
                        )
                        checkpoint = build_checkpoint(
                            args=args,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            model_info=model_info,
                            usable_rows=usable_rows,
                            next_parameter_id=next_parameter_id,
                            pool_cycle=pool_cycle,
                            global_step=global_step,
                            total_samples=total_samples,
                            total_elapsed_seconds=elapsed_total,
                            best_score=best_score,
                        )
                        atomic_torch_save(checkpoint, latest_path)
                        print(f"已保存 latest checkpoint: {latest_path}")

                del target_scaled, g_clean_scaled
                block_offset = chunk_end

    except KeyboardInterrupt:
        stop_reason = "用户按下 Ctrl+C"
        print("\n收到 Ctrl+C，正在安全保存检查点……")

    run_elapsed = time.monotonic() - start_time
    total_elapsed = previous_elapsed_seconds + run_elapsed

    if not math.isfinite(best_score):
        fallback_score = float(np.mean(loss_window)) if loss_window else float("inf")
        best_score = fallback_score
        save_best_model(
            checkpoint_dir=checkpoint_dir,
            model=model,
            model_info=model_info,
            score=best_score,
            global_step=global_step,
            pool_cycle=pool_cycle,
            next_parameter_id=next_parameter_id,
        )

    final_checkpoint = build_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_info=model_info,
        usable_rows=usable_rows,
        next_parameter_id=next_parameter_id,
        pool_cycle=pool_cycle,
        global_step=global_step,
        total_samples=total_samples,
        total_elapsed_seconds=total_elapsed,
        best_score=best_score,
    )
    atomic_torch_save(final_checkpoint, latest_path)

    throughput = run_samples / max(run_elapsed, 1e-9)
    full_pool_hours = (
        usable_rows / throughput / 3600.0
        if throughput > 0
        else float("inf")
    )
    target_160m_hours = (
        160_000_000 / throughput / 3600.0
        if throughput > 0
        else float("inf")
    )

    summary = {
        **model_info,
        "stop_reason": stop_reason or "训练循环正常结束",
        "run_steps": global_step - start_global_step,
        "run_samples": int(run_samples),
        "run_elapsed_seconds": float(run_elapsed),
        "run_throughput_samples_per_second": float(throughput),
        "estimated_current_pool_hours": float(full_pool_hours),
        "estimated_160m_hours": float(target_160m_hours),
        "pool_usable_rows": int(usable_rows),
        "pool_cycle": int(pool_cycle),
        "next_parameter_id": int(next_parameter_id),
        "global_step": int(global_step),
        "total_samples": int(total_samples),
        "total_elapsed_seconds": float(total_elapsed),
        "best_score": float(best_score),
        "latest_checkpoint": str(latest_path.resolve()),
        "best_model": str((checkpoint_dir / "best_model.pth").resolve()),
    }
    atomic_json_save(summary, checkpoint_dir / "training_summary.json")

    print("=" * 72)
    print(f"停止原因: {summary['stop_reason']}")
    print(f"本次步数: {summary['run_steps']:,}")
    print(f"本次样本: {run_samples:,}")
    print(f"本次耗时: {run_elapsed / 3600.0:.3f} 小时")
    print(f"实际吞吐: {throughput:,.2f} samples/s")
    print(f"覆盖当前参数池一次预计: {full_pool_hours:,.2f} 小时")
    print(f"覆盖 1.6 亿参数一次预计: {target_160m_hours:,.2f} 小时")
    print(f"latest: {latest_path.resolve()}")
    print(f"best: {(checkpoint_dir / 'best_model.pth').resolve()}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
