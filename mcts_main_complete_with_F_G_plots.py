from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data_generate import compute_gy
from pick_one_m_not_065 import find_best_sample, save_one_sample
from TransformerInverse import InverseTransformer1D
from mcts_refinement_rl_prior_v2 import (
    AdaptiveScoreGuidedPriorPolicy,
    MCTSRefinement,
)


# ============================================================
# 基础路径与样本设置
# ============================================================

MODEL_PATH = "./model/two_truth10_transformer_pinn.pth"
DATA_PATH = "./data_two_truth10/test.npz"

# MCTS 和 Transformer 实际使用的输入。
# 带噪声输入使用 "gy_noisy"；无噪声输入使用 "gy_clean"。
GY_KEY = "gy_noisy"

OUTPUT_DIR = "./mcts_result_selected_m_gamma"

# 为了直接生成你截图中的“三个对应样本、两列图”，这里可以填写测试集索引。
# 截图中显示的是 1708、367、52，因此先按这三个索引配置。
#
# 若仍想完全沿用 pick_one_m_not_065.py 的单样本选择方式，
# 将 SAMPLE_INDICES 改为 None 即可。
SAMPLE_INDICES: tuple[int, ...] | None = (367,)# 367, 52)


# ============================================================
# 原来的 MCTS 参数：保持不变
# ============================================================

MCTS_ITERATIONS = 4000
ROLLOUT_DEPTH = 8

SHOW_PLOTS = True
SEED = 0


# ============================================================
# 通用工具
# ============================================================

def set_global_seed(seed: int) -> None:
    """设置外围程序的随机种子；不改变 MCTS 内部参数和搜索逻辑。"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mse(a, b)))


def relative_mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    相对 MSE：

        mean((a - b)^2) / mean(b^2)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return float(
        np.mean((a - b) ** 2)
        / (np.mean(b ** 2) + 1e-12)
    )


def relative_rmse_percent(a: np.ndarray, b: np.ndarray) -> float:
    """返回相对 RMSE 百分比。"""
    return float(100.0 * np.sqrt(relative_mse(a, b)))


def to_jsonable(value: Any) -> Any:
    """
    把 NumPy 标量、数组以及嵌套容器转换成 JSON 可序列化对象。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    return value


def save_figure(
    fig: plt.Figure,
    path: Path,
    show: bool = SHOW_PLOTS,
) -> Path:
    """
    保存指定 Figure，避免依赖全局 plt 当前图。
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)
    return path


# ============================================================
# 数据读取
# ============================================================

def choose_gy_key(data: np.lib.npyio.NpzFile) -> str:
    """
    选择 Transformer 和 MCTS 使用的 g(y) 字段。
    """
    if GY_KEY in data.files:
        return GY_KEY

    for key in (
        "gy_noisy",
        "gy_clean",
        "gy",
    ):
        if key in data.files:
            print(
                f"Warning: requested GY_KEY={GY_KEY!r} "
                f"does not exist; using {key!r} instead."
            )
            return key

    raise KeyError(
        "No g(y) input was found. "
        "Expected one of: gy_noisy, gy_clean, gy. "
        f"Current fields: {data.files}"
    )


def extract_sample_field(
    value: np.ndarray,
    index: int,
    sample_count: int,
    field_name: str,
) -> np.ndarray:
    """
    提取带样本维的字段，例如 fx、gy、m、gamma。

    支持：
        [N, L]
        [N, 1]
        [N]
    """
    array = np.asarray(value)

    if array.ndim == 0:
        return array

    if array.shape[0] != sample_count:
        raise ValueError(
            f"{field_name!r} 的第一维应为样本数 {sample_count}，"
            f"实际形状为 {array.shape}。"
        )

    return np.asarray(array[index])


def extract_grid_field(
    value: np.ndarray,
    index: int,
    sample_count: int,
    field_name: str,
) -> np.ndarray:
    """
    提取 x/y 网格。

    支持公共网格 [L]，也支持逐样本网格 [N, L]。
    与旧版 _take_sample 不同，不会把长度恰好等于样本数的一维网格误判成样本维。
    """
    array = np.asarray(value)

    if array.ndim == 1:
        return array

    if array.ndim >= 2 and array.shape[0] == sample_count:
        return np.asarray(array[index])

    if array.ndim >= 2 and array.shape[0] == 1:
        return np.asarray(array[0])

    raise ValueError(
        f"无法识别 {field_name!r} 网格形状 {array.shape}。"
    )


def load_one_sample_from_npz(
    data: np.lib.npyio.NpzFile,
    index: int,
    split: str,
    source_path: Path,
    gy_key: str,
) -> dict[str, Any]:
    """
    从已经打开的 npz 中读取一个样本。

    fx_true 只用于 MCTS 完成后的评估和画图；
    它不会输入 Transformer，也不会参与 MCTS 搜索。
    """
    fx_all = np.asarray(data["fx"])
    sample_count = int(fx_all.shape[0])

    if index < 0 or index >= sample_count:
        raise IndexError(
            f"样本索引 {index} 超出范围；"
            f"当前数据集共有 {sample_count} 条样本。"
        )

    fx_true = np.asarray(
        extract_sample_field(
            data["fx"],
            index,
            sample_count,
            "fx",
        ),
        dtype=np.float64,
    ).reshape(-1)

    gy_target = np.asarray(
        extract_sample_field(
            data[gy_key],
            index,
            sample_count,
            gy_key,
        ),
        dtype=np.float64,
    ).reshape(-1)

    if "x" in data.files:
        x = np.asarray(
            extract_grid_field(
                data["x"],
                index,
                sample_count,
                "x",
            ),
            dtype=np.float64,
        ).reshape(-1)
    else:
        x = np.linspace(
            0.0,
            2.0,
            len(fx_true),
            dtype=np.float64,
        )

    if "y" in data.files:
        y = np.asarray(
            extract_grid_field(
                data["y"],
                index,
                sample_count,
                "y",
            ),
            dtype=np.float64,
        ).reshape(-1)
    else:
        y = np.linspace(
            3.0,
            8.0,
            len(gy_target),
            dtype=np.float64,
        )

    m_value = float(
        np.asarray(
            extract_sample_field(
                data["m"],
                index,
                sample_count,
                "m",
            )
        ).reshape(-1)[0]
    )

    gamma_value = float(
        np.asarray(
            extract_sample_field(
                data["gamma"],
                index,
                sample_count,
                "gamma",
            )
        ).reshape(-1)[0]
    )

    if len(x) != len(fx_true):
        raise ValueError(
            f"index={index}: x 长度 {len(x)} "
            f"与 fx 长度 {len(fx_true)} 不一致。"
        )

    if len(y) != len(gy_target):
        raise ValueError(
            f"index={index}: y 长度 {len(y)} "
            f"与 {gy_key} 长度 {len(gy_target)} 不一致。"
        )

    # 读取干净 g(y)。没有该字段时，由真实 f(x) 正演得到。
    if "gy_clean" in data.files:
        gy_clean = np.asarray(
            extract_sample_field(
                data["gy_clean"],
                index,
                sample_count,
                "gy_clean",
            ),
            dtype=np.float64,
        ).reshape(-1)
    else:
        gy_clean = np.asarray(
            compute_gy(
                x,
                fx_true,
                y,
            ),
            dtype=np.float64,
        ).reshape(-1)

    # 读取带噪声 g(y)。没有该字段时，退化为当前输入或干净数据。
    if "gy_noisy" in data.files:
        gy_noisy = np.asarray(
            extract_sample_field(
                data["gy_noisy"],
                index,
                sample_count,
                "gy_noisy",
            ),
            dtype=np.float64,
        ).reshape(-1)
    elif gy_key == "gy":
        gy_noisy = gy_target.copy()
    else:
        gy_noisy = gy_clean.copy()

    if len(gy_clean) != len(y):
        raise ValueError(
            f"index={index}: gy_clean 长度 {len(gy_clean)} "
            f"与 y 长度 {len(y)} 不一致。"
        )

    if len(gy_noisy) != len(y):
        raise ValueError(
            f"index={index}: gy_noisy 长度 {len(gy_noisy)} "
            f"与 y 长度 {len(y)} 不一致。"
        )

    return {
        "x": x,
        "y": y,
        "fx_true": fx_true,
        "gy_clean": gy_clean,
        "gy_noisy": gy_noisy,
        "gy_target": gy_target,
        "gy_key": gy_key,
        "m": m_value,
        "gamma": gamma_value,
        "split": split,
        "local_index": int(index),
        "source_path": str(source_path),
    }


def load_requested_samples() -> list[dict[str, Any]]:
    """
    两种运行方式：

    1. SAMPLE_INDICES 非空：
       直接从 DATA_PATH 读取这些测试集索引，用于生成三行两列总览图。

    2. SAMPLE_INDICES 为 None：
       完全沿用 pick_one_m_not_065.py 的 find_best_sample() 逻辑。
    """
    if SAMPLE_INDICES:
        source_path = Path(DATA_PATH)
        split = "test"
        indices = [
            int(index)
            for index in SAMPLE_INDICES
        ]
    else:
        best = find_best_sample()
        save_one_sample(best)

        source_path = Path(best["path"])
        split = str(best["split"])
        indices = [
            int(best["local_idx"])
        ]

    if not source_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件: {source_path}"
        )

    with np.load(
        source_path,
        allow_pickle=True,
    ) as data:
        required = {
            "fx",
            "m",
            "gamma",
        }
        missing = sorted(
            required.difference(data.files)
        )

        if missing:
            raise KeyError(
                f"数据文件缺少字段 {missing}；"
                f"现有字段为 {data.files}。"
            )

        gy_key = choose_gy_key(data)

        samples = [
            load_one_sample_from_npz(
                data=data,
                index=index,
                split=split,
                source_path=source_path,
                gy_key=gy_key,
            )
            for index in indices
        ]

    print()
    print("========== Selected Samples ==========")

    for position, sample in enumerate(
        samples,
        start=1,
    ):
        print(
            f"sample {position}: "
            f"split={sample['split']}, "
            f"index={sample['local_index']}, "
            f"m={sample['m']}, "
            f"gamma={sample['gamma']}, "
            f"m^2={sample['m'] ** 2}, "
            f"gy_key={sample['gy_key']}"
        )

    return samples


def validate_common_grids(
    samples: list[dict[str, Any]],
) -> None:
    """
    当前模型只需加载一次，因此要求这些样本使用相同的 x/y 网格。
    """
    if not samples:
        raise ValueError("没有可运行的样本。")

    reference_x = samples[0]["x"]
    reference_y = samples[0]["y"]

    for sample in samples[1:]:
        if (
            len(sample["x"]) != len(reference_x)
            or not np.allclose(
                sample["x"],
                reference_x,
            )
        ):
            raise ValueError(
                "多个样本的 x 网格不一致，"
                "无法安全地共用同一个已训练模型。"
            )

        if (
            len(sample["y"]) != len(reference_y)
            or not np.allclose(
                sample["y"],
                reference_y,
            )
        ):
            raise ValueError(
                "多个样本的 y 网格不一致，"
                "无法安全地共用同一个已训练模型。"
            )


# ============================================================
# Transformer 初始模型
# ============================================================

def unwrap_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    支持：
        1. 直接保存的 state_dict
        2. {"state_dict": ...}
        3. {"model_state_dict": ...}

    同时兼容 DataParallel 的 "module." 前缀。
    """
    state = checkpoint

    if isinstance(state, dict):
        for key in (
            "state_dict",
            "model_state_dict",
        ):
            if (
                key in state
                and isinstance(state[key], dict)
            ):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError(
            "权重文件不是可识别的 PyTorch state_dict。"
        )

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        new_key = str(key)

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_real_inverse_model(
    x: np.ndarray,
    y: np.ndarray,
    model_path: str = MODEL_PATH,
) -> tuple[InverseTransformer1D, torch.device]:
    """
    保留你原来正确的 Transformer 建模方式：
    输入长度、输出长度和坐标范围直接由实际 x/y 网格确定。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到 Transformer 权重: {model_path}"
        )

    cfg = Config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # 这里保持原模型结构和参数来源不变。
    model = InverseTransformer1D(
        input_length=len(y),
        output_length=len(x),
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_encoder_layers=(
            cfg.transformer_num_layers
        ),
        num_decoder_layers=(
            cfg.transformer_num_layers
        ),
        dim_feedforward=(
            cfg.transformer_dim_feedforward
        ),
        dropout=cfg.transformer_dropout,
        x_min=float(x[0]),
        x_max=float(x[-1]),
        y_min=float(y[0]),
        y_max=float(y[-1]),
    ).to(device)

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print()
    print("========== Initial Transformer ==========")
    print("checkpoint :", model_path)
    print("device     :", device)
    print("input len  :", len(y))
    print("output len :", len(x))

    return model, device


def predict_initial_fx(
    model: InverseTransformer1D,
    device: torch.device,
    gy_target: np.ndarray,
) -> np.ndarray:
    """
    只把 g(y) 输入 Transformer，得到 MCTS 初始 f(x)。
    """
    gy_tensor = torch.from_numpy(
        np.asarray(
            gy_target,
            dtype=np.float32,
        )
    ).view(
        1,
        1,
        -1,
    ).to(device)

    with torch.no_grad():
        prediction = model(
            gy_tensor
        )

    fx_init = np.asarray(
        prediction
        .squeeze(0)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy(),
        dtype=np.float64,
    ).reshape(-1)

    # 保持与你原来 MCTS project() 一致的非负约束。
    return np.clip(
        fx_init,
        0.0,
        None,
    )


# ============================================================
# 原来的 V2 policy 与 MCTS：参数和逻辑保持不变
# ============================================================

def build_original_v2_policy() -> AdaptiveScoreGuidedPriorPolicy:
    return AdaptiveScoreGuidedPriorPolicy(
        num_candidates=48,
        structured_fraction=0.60,
        temperature=0.45,
        min_prior=0.08,
        max_prior=3.0,
        roughness_weight=0.50,
        prior_deviation_weight=0.30,
    )


def build_original_v2_mcts(
    x: np.ndarray,
    y: np.ndarray,
    gy_target: np.ndarray,
    policy: AdaptiveScoreGuidedPriorPolicy,
    seed: int,
) -> MCTSRefinement:
    return MCTSRefinement(
        x=x,
        y=y,
        gy_target=gy_target,

        iterations=MCTS_ITERATIONS,
        rollout_depth=ROLLOUT_DEPTH,

        lambda_tv=0.0045,
        lambda_curv=0.00060,
        lambda_prior=0.100,
        lambda_mass=0.025,

        exploration=1.00,
        max_children=32,
        progressive_c=2.0,
        progressive_alpha=0.50,

        peak_amp_range=(
            0.00001,
            0.003,
        ),

        peak_width_range=(
            0.010,
            0.060,
        ),

        policy_fn=policy,
        policy_prior_weight=1.0,
        prior_floor=0.08,
        prior_ceiling=3.0,

        restart_patience=300,
        min_improvement=1e-4,

        polish_smoothing_sigmas=(
            # 0.25,
            # 0.50,
            # 1.00,
        ),

        random_state=seed + 100,
    )


# ============================================================
# 单样本绘图
# ============================================================

def plot_f_reconstruction(
    result: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    与截图左列对应：
    True f(x) 与最终 MCTS Predicted f(x)，并标出 m^2。
    """
    sample = result["sample"]
    metrics = result["metrics"]

    fig, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    axis.plot(
        sample["x"],
        sample["fx_true"],
        label="True f(x)",
        linewidth=2.0,
    )

    axis.plot(
        sample["x"],
        result["fx_refined"],
        linestyle="--",
        label="Predicted f(x) after MCTS",
        linewidth=2.0,
    )

    m_squared = sample["m"] ** 2

    if (
        sample["x"][0]
        <= m_squared
        <= sample["x"][-1]
    ):
        axis.axvline(
            m_squared,
            linestyle=":",
            linewidth=1.2,
            label=f"m^2={m_squared:.3f}",
        )

    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_title(
        f"F reconstruction | "
        f"m={sample['m']:.3f}, "
        f"gamma={sample['gamma']:.3f}\n"
        f"RMSE={metrics['formal_true_rmse']:.6f}"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    return save_figure(
        fig,
        output_path,
    )


def plot_g_noise_comparison(
    result: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    与截图右列对应：
    Clean g(y) 与 noisy g(y)。
    """
    sample = result["sample"]
    metrics = result["metrics"]

    fig, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    axis.plot(
        sample["y"],
        sample["gy_clean"],
        label="Clean g(y)",
        linewidth=2.0,
    )

    axis.plot(
        sample["y"],
        sample["gy_noisy"],
        linestyle="--",
        label="Noisy g(y)",
        linewidth=1.8,
    )

    axis.set_xlabel("y")
    axis.set_ylabel("g(y)")
    axis.set_title(
        f"G noise comparison | "
        f"same index={sample['local_index']}\n"
        f"RMSE={metrics['g_noise_rmse']:.6f}, "
        f"relative={metrics['g_noise_relative_rmse_percent']:.2f}%"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    return save_figure(
        fig,
        output_path,
    )


def plot_f_refinement_detail(
    result: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    额外保留 Transformer 初值和 MCTS 结果，便于看 MCTS 是否真正改善。
    """
    sample = result["sample"]

    fig, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        sample["x"],
        sample["fx_true"],
        label="True f(x)",
        linewidth=2.5,
    )

    axis.plot(
        sample["x"],
        result["fx_init"],
        linestyle="--",
        label="Transformer Initial f(x)",
        linewidth=1.8,
    )

    axis.plot(
        sample["x"],
        result["fx_refined"],
        label="MCTS Refined f(x)",
        linewidth=1.8,
    )

    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_title(
        "Transformer Initial Guess and MCTS Refinement"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    return save_figure(
        fig,
        output_path,
    )


def plot_forward_consistency(
    result: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    保留原来的正演一致性图。
    """
    sample = result["sample"]

    fig, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        sample["y"],
        sample["gy_target"],
        label=f"Target g(y): {sample['gy_key']}",
        linewidth=2.5,
    )

    axis.plot(
        sample["y"],
        result["gy_init"],
        linestyle="--",
        label="Transformer Initial g(y)",
        linewidth=1.8,
    )

    axis.plot(
        sample["y"],
        result["gy_refined"],
        label="MCTS Refined g(y)",
        linewidth=1.8,
    )

    axis.set_xlabel("y")
    axis.set_ylabel("g(y)")
    axis.set_title("Forward Consistency")
    axis.grid(alpha=0.25)
    axis.legend()

    return save_figure(
        fig,
        output_path,
    )


def add_restart_lines(
    axis: plt.Axes,
    restart_steps: list[int],
) -> None:
    for position, restart_step in enumerate(
        restart_steps
    ):
        axis.axvline(
            restart_step,
            linestyle=":",
            linewidth=1.0,
            label=(
                "Tree Restart"
                if position == 0
                else None
            ),
        )


def plot_mcts_histories(
    info: dict[str, Any],
    sample_dir: Path,
) -> list[Path]:
    """
    保留原 main 中的三张 MCTS 诊断图。
    """
    history = info["history"]
    restart_steps = [
        int(value)
        for value in info.get(
            "restart_steps",
            [],
        )
    ]

    paths: list[Path] = []

    fig, axis = plt.subplots(
        figsize=(10, 5)
    )
    axis.plot(
        history["best_score"],
        label="Best Score",
    )
    add_restart_lines(
        axis,
        restart_steps,
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Score")
    axis.set_title("MCTS Optimization History")
    axis.grid(alpha=0.25)
    axis.legend()
    paths.append(
        save_figure(
            fig,
            sample_dir / "05_mcts_score_history.png",
        )
    )

    fig, axis = plt.subplots(
        figsize=(10, 5)
    )
    axis.plot(
        history["root_prior_min"],
        label="Root Prior Min",
    )
    axis.plot(
        history["root_prior_mean"],
        label="Root Prior Mean",
    )
    axis.plot(
        history["root_prior_max"],
        label="Root Prior Max",
    )
    add_restart_lines(
        axis,
        restart_steps,
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Prior")
    axis.set_title("Policy Prior Statistics")
    axis.grid(alpha=0.25)
    axis.legend()
    paths.append(
        save_figure(
            fig,
            sample_dir / "06_policy_prior_statistics.png",
        )
    )

    fig, axis = plt.subplots(
        figsize=(10, 5)
    )
    axis.plot(
        history["root_num_children"],
        label="Root Children",
    )
    add_restart_lines(
        axis,
        restart_steps,
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Count")
    axis.set_title(
        "Root Children / Progressive Widening"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    paths.append(
        save_figure(
            fig,
            sample_dir / "07_root_children.png",
        )
    )

    return paths


# ============================================================
# 单个样本运行
# ============================================================

def run_single_sample(
    sample: dict[str, Any],
    model: InverseTransformer1D,
    device: torch.device,
    seed: int,
    ordinal: int,
) -> dict[str, Any]:
    """
    单个样本的执行顺序保持为：

        noisy/clean g(y)
        -> Transformer 初值
        -> 原 V2 policy
        -> 原 V2 MCTS refine
        -> 正演与事后评估
    """
    sample_dir = (
        Path(OUTPUT_DIR)
        / (
            f"sample_{ordinal:02d}_"
            f"index_{sample['local_index']}"
        )
    )
    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = sample["x"]
    y = sample["y"]
    fx_true = sample["fx_true"]
    gy_target = sample["gy_target"]

    # 1. Transformer 初值。
    fx_init = predict_initial_fx(
        model=model,
        device=device,
        gy_target=gy_target,
    )

    if len(fx_init) != len(x):
        raise ValueError(
            f"Transformer 输出长度 {len(fx_init)} "
            f"与 x 长度 {len(x)} 不一致。"
        )

    # 2. 原 V2 policy。
    policy = build_original_v2_policy()

    # 3. 原 V2 MCTS。
    refiner = build_original_v2_mcts(
        x=x,
        y=y,
        gy_target=gy_target,
        policy=policy,
        seed=seed,
    )

    fx_refined, info = refiner.refine(
        fx_init,
        verbose=True,
        return_info=True,
    )

    fx_refined = np.asarray(
        fx_refined,
        dtype=np.float64,
    ).reshape(-1)

    # 4. 正演。
    gy_init = np.asarray(
        compute_gy(
            x,
            fx_init,
            y,
        ),
        dtype=np.float64,
    ).reshape(-1)

    gy_refined = np.asarray(
        compute_gy(
            x,
            fx_refined,
            y,
        ),
        dtype=np.float64,
    ).reshape(-1)

    # 5. 事后评估。
    # fx_true 不参与 Transformer 输入，也不参与 MCTS score。
    formal_true_mse = mse(
        fx_refined,
        fx_true,
    )
    formal_true_rmse = rmse(
        fx_refined,
        fx_true,
    )

    metrics: dict[str, Any] = {
        "split": sample["split"],
        "local_index": int(
            sample["local_index"]
        ),
        "source_path": sample["source_path"],
        "m": float(sample["m"]),
        "gamma": float(sample["gamma"]),
        "m_squared": float(
            sample["m"] ** 2
        ),
        "gy_key": sample["gy_key"],
        "model_path": MODEL_PATH,

        # 保留旧字段名，方便已有下游代码继续使用。
        "formal_true_mse": formal_true_mse,
        "formal_true_rmse": formal_true_rmse,

        # f(x) 初值与最终结果。
        "fx_initial_mse": mse(
            fx_init,
            fx_true,
        ),
        "fx_initial_rmse": rmse(
            fx_init,
            fx_true,
        ),
        "fx_refined_mse": formal_true_mse,
        "fx_refined_rmse": formal_true_rmse,
        "fx_relative_mse_before": relative_mse(
            fx_init,
            fx_true,
        ),
        "fx_relative_mse_after": relative_mse(
            fx_refined,
            fx_true,
        ),

        # clean/noisy g(y) 的噪声指标。
        "g_noise_mse": mse(
            sample["gy_noisy"],
            sample["gy_clean"],
        ),
        "g_noise_rmse": rmse(
            sample["gy_noisy"],
            sample["gy_clean"],
        ),
        "g_noise_relative_mse": relative_mse(
            sample["gy_noisy"],
            sample["gy_clean"],
        ),
        "g_noise_relative_rmse_percent": (
            relative_rmse_percent(
                sample["gy_noisy"],
                sample["gy_clean"],
            )
        ),

        # 正演相对当前 MCTS 目标的误差。
        "gy_target_mse_before": mse(
            gy_init,
            gy_target,
        ),
        "gy_target_mse_after": mse(
            gy_refined,
            gy_target,
        ),
        "gy_target_rmse_before": rmse(
            gy_init,
            gy_target,
        ),
        "gy_target_rmse_after": rmse(
            gy_refined,
            gy_target,
        ),
        "gy_relative_mse_before": relative_mse(
            gy_init,
            gy_target,
        ),
        "gy_relative_mse_after": relative_mse(
            gy_refined,
            gy_target,
        ),

        # 正演相对干净真值的误差，仅用于事后观察。
        "gy_clean_rmse_before": rmse(
            gy_init,
            sample["gy_clean"],
        ),
        "gy_clean_rmse_after": rmse(
            gy_refined,
            sample["gy_clean"],
        ),

        "mcts_best_score": float(
            info["best_score"]
        ),
        "root_prior_stats": to_jsonable(
            info.get(
                "root_prior_stats",
                {},
            )
        ),
        "restart_steps": [
            int(value)
            for value in info.get(
                "restart_steps",
                [],
            )
        ],
    }

    print()
    print(
        "========== Final Metrics | "
        f"index={sample['local_index']} =========="
    )
    print(
        "f initial RMSE         :",
        f"{metrics['fx_initial_rmse']:.9e}",
    )
    print(
        "f MCTS RMSE            :",
        f"{metrics['fx_refined_rmse']:.9e}",
    )
    print(
        "g noise relative RMSE  :",
        f"{metrics['g_noise_relative_rmse_percent']:.4f}%",
    )
    print(
        "g target RMSE before   :",
        f"{metrics['gy_target_rmse_before']:.9e}",
    )
    print(
        "g target RMSE after    :",
        f"{metrics['gy_target_rmse_after']:.9e}",
    )
    print(
        "MCTS best score        :",
        metrics["mcts_best_score"],
    )
    print(
        "restart steps          :",
        metrics["restart_steps"],
    )

    # 6. 保存指标。
    metrics_path = (
        sample_dir
        / "mcts_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_jsonable(metrics),
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 7. 保存完整数组。
    result_path = (
        sample_dir
        / "mcts_result.npz"
    )

    np.savez_compressed(
        result_path,
        x=x,
        y=y,
        fx_true=fx_true,
        fx_init=fx_init,
        fx_refined=fx_refined,
        gy_clean=sample["gy_clean"],
        gy_noisy=sample["gy_noisy"],
        gy_target=gy_target,
        gy_init=gy_init,
        gy_refined=gy_refined,
        selected_m=np.asarray(
            sample["m"],
            dtype=np.float64,
        ),
        selected_gamma=np.asarray(
            sample["gamma"],
            dtype=np.float64,
        ),
        original_split=np.asarray(
            sample["split"],
        ),
        original_local_index=np.asarray(
            sample["local_index"],
            dtype=np.int64,
        ),
        formal_true_mse=np.asarray(
            formal_true_mse,
            dtype=np.float64,
        ),
        formal_true_rmse=np.asarray(
            formal_true_rmse,
            dtype=np.float64,
        ),
    )

    result = {
        "sample": sample,
        "fx_init": fx_init,
        "fx_refined": fx_refined,
        "gy_init": gy_init,
        "gy_refined": gy_refined,
        "info": info,
        "metrics": metrics,
        "sample_dir": str(sample_dir),
        "metrics_path": str(metrics_path),
        "result_path": str(result_path),
    }

    # 8. 两张用户需要的图。
    f_path = plot_f_reconstruction(
        result,
        sample_dir / "01_f_reconstruction.png",
    )

    g_path = plot_g_noise_comparison(
        result,
        sample_dir / "02_g_noise_comparison.png",
    )

    # 9. 额外保留原 main 的分析图。
    detail_path = plot_f_refinement_detail(
        result,
        sample_dir / "03_f_refinement_detail.png",
    )

    forward_path = plot_forward_consistency(
        result,
        sample_dir / "04_forward_consistency.png",
    )

    history_paths = plot_mcts_histories(
        info,
        sample_dir,
    )

    result["figure_paths"] = [
        str(f_path),
        str(g_path),
        str(detail_path),
        str(forward_path),
        *[
            str(path)
            for path in history_paths
        ],
    ]

    return result


# ============================================================
# 多样本总览图与汇总
# ============================================================

def plot_corresponding_samples_overview(
    results: list[dict[str, Any]],
) -> Path:
    """
    生成与截图结构一致的图：

        左列：F reconstruction
        右列：G noise comparison
        每一行对应同一个样本索引。
    """
    row_count = len(results)

    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(
            16,
            5.2 * row_count,
        ),
        squeeze=False,
    )

    for row, result in enumerate(results):
        sample = result["sample"]
        metrics = result["metrics"]

        left_axis = axes[row, 0]
        right_axis = axes[row, 1]

        # 左：F reconstruction。
        left_axis.plot(
            sample["x"],
            sample["fx_true"],
            label="True f(x)",
            linewidth=2.0,
        )
        left_axis.plot(
            sample["x"],
            result["fx_refined"],
            linestyle="--",
            label="Predicted f(x) after MCTS",
            linewidth=1.8,
        )

        m_squared = sample["m"] ** 2

        if (
            sample["x"][0]
            <= m_squared
            <= sample["x"][-1]
        ):
            left_axis.axvline(
                m_squared,
                linestyle=":",
                linewidth=1.1,
                label=f"m^2={m_squared:.3f}",
            )

        left_axis.set_xlabel("x")
        left_axis.set_ylabel("f(x)")
        left_axis.set_title(
            f"Sample {row + 1}: F | "
            f"m={sample['m']:.3f}, "
            f"gamma={sample['gamma']:.3f}\n"
            f"RMSE={metrics['formal_true_rmse']:.6f}"
        )
        left_axis.grid(alpha=0.25)
        left_axis.legend()

        # 右：G noise comparison。
        right_axis.plot(
            sample["y"],
            sample["gy_clean"],
            label="Clean g(y)",
            linewidth=2.0,
        )
        right_axis.plot(
            sample["y"],
            sample["gy_noisy"],
            linestyle="--",
            label="Noisy g(y)",
            linewidth=1.6,
        )
        right_axis.set_xlabel("y")
        right_axis.set_ylabel("g(y)")
        right_axis.set_title(
            f"Sample {row + 1}: G | "
            f"same index={sample['local_index']}\n"
            f"RMSE={metrics['g_noise_rmse']:.6f}, "
            f"relative="
            f"{metrics['g_noise_relative_rmse_percent']:.2f}%"
        )
        right_axis.grid(alpha=0.25)
        right_axis.legend()

    fig.suptitle(
        "Corresponding samples: "
        "F reconstruction (left) and G noise comparison (right)",
        fontsize=15,
        y=1.005,
    )

    return save_figure(
        fig,
        Path(OUTPUT_DIR)
        / "00_corresponding_samples_F_G.png",
    )


def save_summary(
    results: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """
    保存全部样本的 CSV 和 JSON 汇总。
    """
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for result in results:
        metrics = result["metrics"]

        rows.append(
            {
                "split": metrics["split"],
                "local_index": metrics["local_index"],
                "m": metrics["m"],
                "gamma": metrics["gamma"],
                "m_squared": metrics["m_squared"],
                "gy_key": metrics["gy_key"],
                "fx_initial_rmse": metrics["fx_initial_rmse"],
                "fx_refined_rmse": metrics["fx_refined_rmse"],
                "g_noise_rmse": metrics["g_noise_rmse"],
                "g_noise_relative_rmse_percent": (
                    metrics[
                        "g_noise_relative_rmse_percent"
                    ]
                ),
                "gy_target_rmse_before": (
                    metrics["gy_target_rmse_before"]
                ),
                "gy_target_rmse_after": (
                    metrics["gy_target_rmse_after"]
                ),
                "mcts_best_score": (
                    metrics["mcts_best_score"]
                ),
                "sample_dir": result["sample_dir"],
            }
        )

    csv_path = (
        output_dir
        / "mcts_samples_summary.csv"
    )

    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        output_dir
        / "mcts_samples_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_jsonable(rows),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


# ============================================================
# 主函数
# ============================================================

def main(
    seed: int = SEED,
) -> list[dict[str, Any]]:
    # 兼容没有 np.trapezoid 的旧版 NumPy。
    if not hasattr(
        np,
        "trapezoid",
    ):
        np.trapezoid = np.trapz

    set_global_seed(seed)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_requested_samples()
    validate_common_grids(samples)

    # 模型结构保持原来的正确写法，只加载一次。
    model, device = load_real_inverse_model(
        x=samples[0]["x"],
        y=samples[0]["y"],
        model_path=MODEL_PATH,
    )

    results: list[dict[str, Any]] = []

    for ordinal, sample in enumerate(
        samples,
        start=1,
    ):
        print()
        print(
            "############################################################"
        )
        print(
            f"Running sample {ordinal}/{len(samples)} | "
            f"index={sample['local_index']}"
        )
        print(
            "############################################################"
        )

        result = run_single_sample(
            sample=sample,
            model=model,
            device=device,
            seed=seed,
            ordinal=ordinal,
        )
        results.append(result)

    overview_path = (
        plot_corresponding_samples_overview(
            results
        )
    )

    csv_path, json_path = save_summary(
        results
    )

    print()
    print("========== Saved Results ==========")
    print("overview :", overview_path)
    print("csv      :", csv_path)
    print("json     :", json_path)

    for result in results:
        print(
            "sample   :",
            result["sample_dir"],
        )

    return results


if __name__ == "__main__":
    main(
        seed=SEED,
    )
