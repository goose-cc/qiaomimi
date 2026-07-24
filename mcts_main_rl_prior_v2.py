import json
import os

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
# 这里是通常需要修改的参数
#
# TARGET_M 和 TARGET_GAMMA
# 仍然在 pick_one_m_not_065.py 中修改。
# ============================================================

MODEL_PATH = "./model/exp5_transformer_pinn.pth"

# 使用带噪声的 g(y)。
# 使用无噪声数据时改为 "gy_clean"。
GY_KEY = "gy_noisy"

OUTPUT_DIR = "./mcts_result_selected_m_gamma"

MCTS_ITERATIONS = 4000
ROLLOUT_DEPTH = 18

SHOW_PLOTS = True
SEED = 0


def relative_mse(a, b):
    """
    计算相对 MSE：

        mean((a - b)^2) / mean(b^2)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(
        np.mean((a - b) ** 2)
        / (np.mean(b ** 2) + 1e-12)
    )


def _take_sample(value, idx, sample_count):
    """
    当数组第一维是样本维度时，提取指定样本。
    对于公共的 x、y 网格则保持原样。
    """
    value = np.asarray(value)

    if (
        value.ndim > 0
        and value.shape[0] == sample_count
    ):
        return value[idx]

    return value


def _choose_gy_key(data):
    """
    选择 MCTS 和 Transformer 使用的 g(y) 字段。
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
        "No gy input was found. "
        "Expected one of: gy_noisy, gy_clean, gy. "
        f"Current fields: {data.files}"
    )


def load_selected_sample():
    """
    调用 pick_one_m_not_065.py 中的样本选择逻辑。

    fx_true 只用于程序结束后的误差计算和画图，
    不输入 Transformer，也不参与 MCTS 搜索。
    """
    best = find_best_sample()

    # 保存选中的单样本 npz。
    save_one_sample(best)

    data = np.load(
        best["path"],
        allow_pickle=True,
    )

    idx = int(best["local_idx"])
    sample_count = len(data["fx"])

    gy_key = _choose_gy_key(data)

    fx_true = np.asarray(
        _take_sample(
            data["fx"],
            idx,
            sample_count,
        ),
        dtype=np.float64,
    ).reshape(-1)

    gy_target = np.asarray(
        _take_sample(
            data[gy_key],
            idx,
            sample_count,
        ),
        dtype=np.float64,
    ).reshape(-1)

    if "x" in data.files:
        x = np.asarray(
            _take_sample(
                data["x"],
                idx,
                sample_count,
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
            _take_sample(
                data["y"],
                idx,
                sample_count,
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
            _take_sample(
                data["m"],
                idx,
                sample_count,
            )
        ).reshape(-1)[0]
    )

    gamma_value = float(
        np.asarray(
            _take_sample(
                data["gamma"],
                idx,
                sample_count,
            )
        ).reshape(-1)[0]
    )

    if len(x) != len(fx_true):
        raise ValueError(
            f"x length ({len(x)}) does not match "
            f"fx length ({len(fx_true)})."
        )

    if len(y) != len(gy_target):
        raise ValueError(
            f"y length ({len(y)}) does not match "
            f"gy length ({len(gy_target)})."
        )

    print()
    print("========== Selected Sample ==========")
    print("split       :", best["split"])
    print("local index :", idx)
    print("m           :", m_value)
    print("gamma       :", gamma_value)
    print("m^2         :", m_value ** 2)
    print("gy key      :", gy_key)
    print("x points    :", len(x))
    print("y points    :", len(y))

    return {
        "x": x,
        "y": y,
        "fx_true": fx_true,
        "gy_target": gy_target,
        "gy_key": gy_key,
        "m": m_value,
        "gamma": gamma_value,
        "split": best["split"],
        "local_index": idx,
    }


def _unwrap_state_dict(checkpoint):
    """
    支持以下几种权重格式：

    1. 直接保存的 state_dict
    2. {"state_dict": ...}
    3. {"model_state_dict": ...}
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
            "The checkpoint is not a PyTorch state_dict "
            "or a supported wrapped checkpoint."
        )

    cleaned = {}

    for key, value in state.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_real_inverse_model(
    x,
    y,
    model_path=MODEL_PATH,
):
    """
    加载真实训练好的 exp5 Transformer + PINN 权重。

    这里完全替代 Fake97Model。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Transformer checkpoint does not exist: "
            f"{model_path}\n"
            "Expected the trained file used by main.py, "
            "for example:\n"
            "./model/exp5_transformer_pinn.pth"
        )

    cfg = Config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    state_dict = _unwrap_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print()
    print("========== Real Initial Model ==========")
    print("checkpoint :", model_path)
    print("device     :", device)

    return model, device


def predict_initial_fx(
    model,
    device,
    gy_target,
):
    """
    使用真实 Transformer 根据 g(y) 生成 MCTS 初始谱。
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
        fx_init = (
            model(gy_tensor)
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # 与 MCTS 中 project() 的非负约束保持一致。
    return np.clip(
        fx_init.astype(np.float64),
        0.0,
        None,
    )


def build_original_v2_policy():
    """
    保留原来的 V2 prior 参数。
    """
    return AdaptiveScoreGuidedPriorPolicy(
        num_candidates=48,
        structured_fraction=0.70,
        temperature=0.55,
        min_prior=0.10,
        max_prior=3.0,
        roughness_weight=0.55,
        prior_deviation_weight=0.45,
    )


def build_original_v2_mcts(
    x,
    y,
    gy_target,
    policy,
    seed,
):
    """
    保留原来的 V2 MCTS 参数。
    """
    return MCTSRefinement(
        x=x,
        y=y,
        gy_target=gy_target,

        iterations=MCTS_ITERATIONS,
        rollout_depth=ROLLOUT_DEPTH,

        lambda_tv=0.0045,
        lambda_curv=0.0009,
        lambda_prior=0.035,
        lambda_mass=0.025,

        exploration=1.35,
        max_children=56,
        progressive_c=3.0,
        progressive_alpha=0.50,

        peak_amp_range=(
            0.00003,
            0.008,
        ),

        peak_width_range=(
            0.004,
            0.055,
        ),

        policy_fn=policy,
        policy_prior_weight=1.0,
        prior_floor=0.10,
        prior_ceiling=3.0,

        restart_patience=300,
        min_improvement=1e-4,

        polish_smoothing_sigmas=(
            0.25,
            0.50,
            #1.00,
        ),

        random_state=seed + 100,
    )


def save_figure(filename):
    """
    保存图片，并根据 SHOW_PLOTS 决定是否弹出窗口。
    """
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return path


def run_once(seed=SEED):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. 选择指定 m、gamma 的样本
    # ========================================================

    sample = load_selected_sample()

    x = sample["x"]
    y = sample["y"]

    fx_true = sample["fx_true"]
    gy_target = sample["gy_target"]

    # ========================================================
    # 2. 使用真实 Transformer 生成初始谱
    #
    # 不使用 Fake97Model。
    # 不把 fx_true 输入 Transformer。
    # ========================================================

    inverse_model, device = (
        load_real_inverse_model(
            x,
            y,
        )
    )

    fx_init = predict_initial_fx(
        inverse_model,
        device,
        gy_target,
    )

    # ========================================================
    # 3. 原来的 V2 prior
    # ========================================================

    policy = build_original_v2_policy()

    # ========================================================
    # 4. 原来的 V2 MCTS 后续逻辑
    # ========================================================

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

    # ========================================================
    # 5. 正演结果
    # ========================================================

    gy_init = compute_gy(
        x,
        fx_init,
        y,
    )

    gy_refined = compute_gy(
        x,
        fx_refined,
        y,
    )

    # ========================================================
    # 6. 计算最终普通 MSE 和 RMSE
    #
    # 这里的 fx_true 只在 MCTS 已经选出最终结果之后使用。
    # 它不参与 MCTS 的综合 score，也不参与搜索过程。
    # ========================================================

    formal_true_mse = float(
        np.mean(
            (
                np.asarray(
                    fx_refined,
                    dtype=np.float64,
                )
                -
                np.asarray(
                    fx_true,
                    dtype=np.float64,
                )
            ) ** 2
        )
    )

    formal_true_rmse = float(
        np.sqrt(formal_true_mse)
    )

    # ========================================================
    # 7. 汇总所有指标
    # ========================================================

    metrics = {
        "split": sample["split"],

        "local_index": int(
            sample["local_index"]
        ),

        "m": float(
            sample["m"]
        ),

        "gamma": float(
            sample["gamma"]
        ),

        "m_squared": float(
            sample["m"] ** 2
        ),

        "gy_key": sample["gy_key"],

        "model_path": MODEL_PATH,

        # 普通 MSE 和 RMSE。
        "formal_true_mse": (
            formal_true_mse
        ),

        "formal_true_rmse": (
            formal_true_rmse
        ),

        # 相对误差。
        "fx_relative_mse_before": (
            relative_mse(
                fx_init,
                fx_true,
            )
        ),

        "fx_relative_mse_after": (
            relative_mse(
                fx_refined,
                fx_true,
            )
        ),

        "gy_relative_mse_before": (
            relative_mse(
                gy_init,
                gy_target,
            )
        ),

        "gy_relative_mse_after": (
            relative_mse(
                gy_refined,
                gy_target,
            )
        ),

        "mcts_best_score": float(
            info["best_score"]
        ),

        "root_prior_stats": (
            info["root_prior_stats"]
        ),

        "restart_steps": [
            int(value)
            for value in info["restart_steps"]
        ],
    }

    # ========================================================
    # 8. 终端打印误差
    # ========================================================

    print()
    print("========== Final Metrics ==========")

    print(
        "formal true MSE        :",
        f"{formal_true_mse:.9e}",
    )

    print(
        "formal true RMSE       :",
        f"{formal_true_rmse:.9e}",
    )

    print(
        "fx relative mse before :",
        metrics["fx_relative_mse_before"],
    )

    print(
        "fx relative mse after  :",
        metrics["fx_relative_mse_after"],
    )

    print(
        "gy relative mse before :",
        metrics["gy_relative_mse_before"],
    )

    print(
        "gy relative mse after  :",
        metrics["gy_relative_mse_after"],
    )

    print(
        "MCTS best score        :",
        metrics["mcts_best_score"],
    )

    print(
        "root prior stats       :",
        metrics["root_prior_stats"],
    )

    print(
        "restart steps          :",
        metrics["restart_steps"],
    )

    # ========================================================
    # 9. 保存 JSON 指标
    # ========================================================

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "mcts_selected_metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 10. 保存完整 npz 数值结果
    # ========================================================

    result_path = os.path.join(
        OUTPUT_DIR,
        "mcts_selected_result.npz",
    )

    np.savez_compressed(
        result_path,

        x=x,
        y=y,

        fx_true=fx_true,
        fx_init=fx_init,
        fx_refined=fx_refined,

        gy_target=gy_target,
        gy_init=gy_init,
        gy_refined=gy_refined,

        formal_true_mse=np.asarray(
            formal_true_mse,
            dtype=np.float64,
        ),

        formal_true_rmse=np.asarray(
            formal_true_rmse,
            dtype=np.float64,
        ),

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
    )

    # ========================================================
    # 11. 真实谱、Transformer 初值和 MCTS 结果图
    #
    # 图标题中显示普通 MSE。
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        fx_true,
        label="True Spectrum",
        linewidth=3,
    )

    plt.plot(
        x,
        fx_init,
        label="Transformer Initial Guess",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        x,
        fx_refined,
        label="V2 Policy-Prior MCTS Refined",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.title(
        "Spectrum Refinement | "
        f"Formal True MSE = {formal_true_mse:.3e}"
    )

    plt.legend()

    spectrum_path = save_figure(
        "01_spectrum_refinement.png"
    )

    # ========================================================
    # 12. 正演一致性图
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        y,
        gy_target,
        label=(
            f"Target g(y): "
            f"{sample['gy_key']}"
        ),
        linewidth=3,
    )

    plt.plot(
        y,
        gy_init,
        label="Transformer Initial g(y)",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        y,
        gy_refined,
        label="MCTS Refined g(y)",
        linewidth=2,
    )

    plt.xlabel("y")
    plt.ylabel("g(y)")
    plt.title("Forward Consistency")
    plt.legend()

    forward_path = save_figure(
        "02_forward_consistency.png"
    )

    # ========================================================
    # 13. MCTS 综合 score 历史
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        info["history"]["best_score"],
        label="Best Score",
    )

    for index, restart_step in enumerate(
        info["restart_steps"]
    ):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label=(
                "Tree Restart"
                if index == 0
                else None
            ),
        )

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("MCTS Optimization History")
    plt.legend()

    score_path = save_figure(
        "03_mcts_score_history.png"
    )

    # ========================================================
    # 14. V2 prior 统计图
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        info["history"]["root_prior_min"],
        label="Root Prior Min",
    )

    plt.plot(
        info["history"]["root_prior_mean"],
        label="Root Prior Mean",
    )

    plt.plot(
        info["history"]["root_prior_max"],
        label="Root Prior Max",
    )

    for index, restart_step in enumerate(
        info["restart_steps"]
    ):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label=(
                "Tree Restart"
                if index == 0
                else None
            ),
        )

    plt.xlabel("Iteration")
    plt.ylabel("Prior")
    plt.title("Policy Prior Statistics")
    plt.legend()

    prior_path = save_figure(
        "04_policy_prior_statistics.png"
    )

    # ========================================================
    # 15. Progressive widening 子节点数量图
    # ========================================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        info["history"]["root_num_children"],
        label="Root Children",
    )

    for index, restart_step in enumerate(
        info["restart_steps"]
    ):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label=(
                "Tree Restart"
                if index == 0
                else None
            ),
        )

    plt.xlabel("Iteration")
    plt.ylabel("Count")

    plt.title(
        "Root Children / Progressive Widening"
    )

    plt.legend()

    children_path = save_figure(
        "05_root_children.png"
    )

    # ========================================================
    # 16. 打印保存位置
    # ========================================================

    print()
    print("========== Saved Results ==========")

    print(result_path)
    print(metrics_path)
    print(spectrum_path)
    print(forward_path)
    print(score_path)
    print(prior_path)
    print(children_path)

    return {
        "x": x,
        "y": y,

        "fx_true": fx_true,
        "fx_init": fx_init,
        "fx_refined": fx_refined,

        "gy_target": gy_target,
        "gy_init": gy_init,
        "gy_refined": gy_refined,

        "formal_true_mse": formal_true_mse,
        "formal_true_rmse": formal_true_rmse,

        "info": info,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # 兼容没有 np.trapezoid 的旧版 NumPy。
    if not hasattr(
        np,
        "trapezoid",
    ):
        np.trapezoid = np.trapz

    run_once(
        seed=SEED
    )
