import numpy as np
import matplotlib.pyplot as plt

from data_generate import generate_fx, compute_gy
from fake_97_model import Fake97Model
from mcts_refinement import MCTSRefinement


def relative_mse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.mean((a - b) ** 2) / (
        np.mean(b ** 2) + 1e-12
    )


def run_once(seed=0):
    x = np.linspace(0.0, 2.0, 256)
    y = np.linspace(2.1, 10.0, 256)

    # 1. 生成真实谱函数
    fx_true = generate_fx(x)

    # 2. forward model 生成观测数据
    gy_true = compute_gy(
        x,
        fx_true,
        y,
    )

    # 3. 生成 synthetic 97% 初始解
    fake_model = Fake97Model(
        target_quality=0.88,
        quality_jitter=0.03,
        random_state=seed,
    )

    fx_init = fake_model.predict(
        fx_true,
        x=x,
    )

    # 4. MCTS refinement
    
    refiner = MCTSRefinement(
        x=x,
        y=y,
        gy_target=gy_true,
        iterations=2500,
        rollout_depth=10,

        # 平滑惩罚降低，避免主峰被压平
        lambda_tv=0.003,
        lambda_curv=0.0005,

        # prior 降低，让 MCTS 可以离开橘线
        lambda_prior=0.04,
        lambda_mass=0.06,

        exploration=2.5,
        max_children=40,
        progressive_c=3.5,
        progressive_alpha=0.55,

        # 幅度别太大，避免乱跳
        peak_amp_range=(0.0003, 0.018),

        # 稍微放宽宽度，让 shift 能覆盖主峰区域
        peak_width_range=(0.006, 0.08),

        random_state=seed + 100,
    )

    fx_refined, info = refiner.refine(
        fx_init,
        verbose=True,
        return_info=True,
    )

    # 5. 计算 forward 结果
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

    # 6. 打印指标
    print()
    print("========== Final Metrics ==========")
    print("fx relative mse before:", relative_mse(fx_init, fx_true))
    print("fx relative mse after :", relative_mse(fx_refined, fx_true))
    print("gy relative mse before:", relative_mse(gy_init, gy_true))
    print("gy relative mse after :", relative_mse(gy_refined, gy_true))
    print("MCTS best score       :", info["best_score"])

    # 7. 可视化 fx
    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        fx_true,
        label="True Spectrum",
        linewidth=3,
    )

    plt.plot(
        x,
        fx_init,
        label="Synthetic 97% Initial Guess",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        x,
        fx_refined,
        label="MCTS Refined",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Spectrum Refinement")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8. 可视化 gy
    plt.figure(figsize=(10, 6))

    plt.plot(
        y,
        gy_true,
        label="Target g(y)",
        linewidth=3,
    )

    plt.plot(
        y,
        gy_init,
        label="Initial g(y)",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        y,
        gy_refined,
        label="Refined g(y)",
        linewidth=2,
    )

    plt.xlabel("y")
    plt.ylabel("g(y)")
    plt.title("Forward Consistency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 9. 可视化 MCTS score
    plt.figure(figsize=(10, 5))

    plt.plot(
        info["history"]["best_score"],
        label="Best Score",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("MCTS Optimization History")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "x": x,
        "y": y,
        "fx_true": fx_true,
        "fx_init": fx_init,
        "fx_refined": fx_refined,
        "gy_true": gy_true,
        "gy_init": gy_init,
        "gy_refined": gy_refined,
        "info": info,
    }


if __name__ == "__main__":
    run_once(seed=0)
