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
# Only these settings normally need to be changed.
#
# TARGET_M and TARGET_GAMMA remain in pick_one_m_not_065.py.
# This script uses the sample selected by that file.
# ============================================================
MODEL_PATH = "./model/exp5_transformer_pinn.pth"

# Keep this the same as the input used by the trained inverse model.
# PeakInversionDataset prioritizes gy_noisy when it exists.
GY_KEY = "gy_noisy"

OUTPUT_DIR = "./mcts_result_selected_m_gamma"
MCTS_ITERATIONS = 50
ROLLOUT_DEPTH = 3
SHOW_PLOTS = True
SEED = 0


def relative_mse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(
        np.mean((a - b) ** 2) /
        (np.mean(b ** 2) + 1e-12)
    )


def _take_sample(value, idx, sample_count):
    """Take one sample only when the first dimension is the sample dimension."""
    value = np.asarray(value)

    if value.ndim > 0 and value.shape[0] == sample_count:
        return value[idx]

    return value


def _choose_gy_key(data):
    if GY_KEY in data.files:
        return GY_KEY

    for key in ("gy_noisy", "gy_clean", "gy"):
        if key in data.files:
            print(
                f"Warning: requested GY_KEY={GY_KEY!r} does not exist; "
                f"using {key!r} instead."
            )
            return key

    raise KeyError(
        "No gy input was found. Expected one of: gy_noisy, gy_clean, gy. "
        f"Current fields: {data.files}"
    )


def load_selected_sample():
    """
    Reuse the selection logic from pick_one_m_not_065.py.

    fx_true is loaded only for reporting/plotting.
    Neither the Transformer inference nor MCTS uses fx_true as an input.
    """
    best = find_best_sample()

    # Preserve the original behavior of pick_one_m_not_065.py:
    # save the selected one-sample npz as well.
    save_one_sample(best)

    data = np.load(best["path"], allow_pickle=True)
    idx = int(best["local_idx"])
    sample_count = len(data["fx"])

    gy_key = _choose_gy_key(data)

    fx_true = np.asarray(
        _take_sample(data["fx"], idx, sample_count),
        dtype=np.float64,
    ).reshape(-1)

    gy_target = np.asarray(
        _take_sample(data[gy_key], idx, sample_count),
        dtype=np.float64,
    ).reshape(-1)

    if "x" in data.files:
        x = np.asarray(
            _take_sample(data["x"], idx, sample_count),
            dtype=np.float64,
        ).reshape(-1)
    else:
        x = np.linspace(0.0, 2.0, len(fx_true), dtype=np.float64)

    if "y" in data.files:
        y = np.asarray(
            _take_sample(data["y"], idx, sample_count),
            dtype=np.float64,
        ).reshape(-1)
    else:
        y = np.linspace(3.0, 8.0, len(gy_target), dtype=np.float64)

    m_value = float(
        np.asarray(_take_sample(data["m"], idx, sample_count)).reshape(-1)[0]
    )
    gamma_value = float(
        np.asarray(_take_sample(data["gamma"], idx, sample_count)).reshape(-1)[0]
    )

    if len(x) != len(fx_true):
        raise ValueError(
            f"x length ({len(x)}) does not match fx length ({len(fx_true)})."
        )

    if len(y) != len(gy_target):
        raise ValueError(
            f"y length ({len(y)}) does not match gy length ({len(gy_target)})."
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
    """Support a raw state_dict and common wrapped checkpoint formats."""
    state = checkpoint

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError(
            "The checkpoint is not a PyTorch state_dict or a supported "
            "wrapped checkpoint."
        )

    cleaned = {}
    for key, value in state.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Some training wrappers save keys as model.xxx.
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_real_inverse_model(x, y, model_path=MODEL_PATH):
    """
    Load the real exp5 Transformer inverse model.

    This replaces Fake97Model completely.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Transformer checkpoint does not exist: {model_path}\n"
            "Expected the trained file used by main.py, for example:\n"
            "./model/exp5_transformer_pinn.pth"
        )

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InverseTransformer1D(
        input_length=len(y),
        output_length=len(x),
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_encoder_layers=cfg.transformer_num_layers,
        num_decoder_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_dim_feedforward,
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
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = _unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print()
    print("========== Real Initial Model ==========")
    print("checkpoint :", model_path)
    print("device     :", device)

    return model, device


def predict_initial_fx(model, device, gy_target):
    gy_tensor = torch.from_numpy(
        np.asarray(gy_target, dtype=np.float32)
    ).view(1, 1, -1).to(device)

    with torch.no_grad():
        fx_init = model(gy_tensor).squeeze(0).squeeze(0).cpu().numpy()

    # The original MCTS enforces a non-negative spectrum in project().
    # Applying the same physical constraint before reporting gy_init keeps
    # "before MCTS" and the actual MCTS root state consistent.
    return np.clip(fx_init.astype(np.float64), 0.0, None)


def build_original_v2_policy():
    """The original V2 prior settings are kept unchanged."""
    return AdaptiveScoreGuidedPriorPolicy(
        num_candidates=36,
        structured_fraction=0.60,
        temperature=0.70,
        min_prior=0.20,
        max_prior=4.0,
        roughness_weight=0.65,
        prior_deviation_weight=0.35,
    )


def build_original_v2_mcts(x, y, gy_target, policy, seed):
    """The original V2 MCTS settings are kept unchanged."""
    return MCTSRefinement(
        x=x,
        y=y,
        gy_target=gy_target,
        iterations=MCTS_ITERATIONS,
        rollout_depth=ROLLOUT_DEPTH,

        lambda_tv=0.006,
        lambda_curv=0.0015,
        lambda_prior=0.055,
        lambda_mass=0.075,

        exploration=1.8,
        max_children=60,
        progressive_c=4.0,
        progressive_alpha=0.55,

        peak_amp_range=(0.00015, 0.012),
        peak_width_range=(0.006, 0.075),

        policy_fn=policy,
        policy_prior_weight=1.0,
        prior_floor=0.20,
        prior_ceiling=4.0,

        restart_patience=450,
        min_improvement=1e-12,
        polish_smoothing_sigmas=(0.35, 0.70, 1.00),

        random_state=seed + 100,
    )


def save_figure(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    return path


def run_once(seed=SEED):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Select m/gamma sample with pick_one_m_not_065.py.
    sample = load_selected_sample()
    x = sample["x"]
    y = sample["y"]
    fx_true = sample["fx_true"]
    gy_target = sample["gy_target"]

    # 2. Real Transformer initial inverse result.
    #    No Fake97Model and no fx_true input.
    inverse_model, device = load_real_inverse_model(x, y)
    fx_init = predict_initial_fx(inverse_model, device, gy_target)

    # 3. Original V2 prior.
    policy = build_original_v2_policy()

    # 4. Original V2 MCTS downstream logic.
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

    # 5. Forward results.
    gy_init = compute_gy(x, fx_init, y)
    gy_refined = compute_gy(x, fx_refined, y)

    # 6. Metrics.
    metrics = {
        "split": sample["split"],
        "local_index": int(sample["local_index"]),
        "m": float(sample["m"]),
        "gamma": float(sample["gamma"]),
        "m_squared": float(sample["m"] ** 2),
        "gy_key": sample["gy_key"],
        "model_path": MODEL_PATH,
        "fx_relative_mse_before": relative_mse(fx_init, fx_true),
        "fx_relative_mse_after": relative_mse(fx_refined, fx_true),
        "gy_relative_mse_before": relative_mse(gy_init, gy_target),
        "gy_relative_mse_after": relative_mse(gy_refined, gy_target),
        "mcts_best_score": float(info["best_score"]),
        "root_prior_stats": info["root_prior_stats"],
        "restart_steps": [int(v) for v in info["restart_steps"]],
    }

    print()
    print("========== Final Metrics ==========")
    print("fx relative mse before:", metrics["fx_relative_mse_before"])
    print("fx relative mse after :", metrics["fx_relative_mse_after"])
    print("gy relative mse before:", metrics["gy_relative_mse_before"])
    print("gy relative mse after :", metrics["gy_relative_mse_after"])
    print("MCTS best score       :", metrics["mcts_best_score"])
    print("root prior stats      :", metrics["root_prior_stats"])
    print("restart steps         :", metrics["restart_steps"])

    with open(
        os.path.join(OUTPUT_DIR, "mcts_selected_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "mcts_selected_result.npz"),
        x=x,
        y=y,
        fx_true=fx_true,
        fx_init=fx_init,
        fx_refined=fx_refined,
        gy_target=gy_target,
        gy_init=gy_init,
        gy_refined=gy_refined,
        selected_m=np.array(sample["m"]),
        selected_gamma=np.array(sample["gamma"]),
        original_split=np.array(sample["split"]),
        original_local_index=np.array(sample["local_index"], dtype=np.int64),
    )

    # 7. Spectrum plot.
    plt.figure(figsize=(10, 6))
    plt.plot(x, fx_true, label="True Spectrum", linewidth=3)
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
    plt.title("Spectrum Refinement")
    plt.legend()
    spectrum_path = save_figure("01_spectrum_refinement.png")

    # 8. Forward plot.
    plt.figure(figsize=(10, 6))
    plt.plot(
        y,
        gy_target,
        label=f"Target g(y): {sample['gy_key']}",
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
    forward_path = save_figure("02_forward_consistency.png")

    # 9. MCTS score history.
    plt.figure(figsize=(10, 5))
    plt.plot(info["history"]["best_score"], label="Best Score")
    for i, restart_step in enumerate(info["restart_steps"]):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label="Tree Restart" if i == 0 else None,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("MCTS Optimization History")
    plt.legend()
    score_path = save_figure("03_mcts_score_history.png")

    # 10. Original V2 prior statistics.
    plt.figure(figsize=(10, 5))
    plt.plot(info["history"]["root_prior_min"], label="Root Prior Min")
    plt.plot(info["history"]["root_prior_mean"], label="Root Prior Mean")
    plt.plot(info["history"]["root_prior_max"], label="Root Prior Max")
    for i, restart_step in enumerate(info["restart_steps"]):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label="Tree Restart" if i == 0 else None,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Prior")
    plt.title("Policy Prior Statistics")
    plt.legend()
    prior_path = save_figure("04_policy_prior_statistics.png")

    # 11. Original progressive widening diagnostic.
    plt.figure(figsize=(10, 5))
    plt.plot(
        info["history"]["root_num_children"],
        label="Root Children",
    )
    for i, restart_step in enumerate(info["restart_steps"]):
        plt.axvline(
            restart_step,
            linestyle=":",
            linewidth=1,
            label="Tree Restart" if i == 0 else None,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("Root Children / Progressive Widening")
    plt.legend()
    children_path = save_figure("05_root_children.png")

    print()
    print("========== Saved Results ==========")
    print(os.path.join(OUTPUT_DIR, "mcts_selected_result.npz"))
    print(os.path.join(OUTPUT_DIR, "mcts_selected_metrics.json"))
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
        "info": info,
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Compatibility for NumPy versions that do not provide np.trapezoid.
    if not hasattr(np, "trapezoid"):
        np.trapezoid = np.trapz

    run_once(seed=SEED)
