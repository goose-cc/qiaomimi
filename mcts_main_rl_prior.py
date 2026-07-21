import numpy as np
import matplotlib.pyplot as plt

from data_generate import generate_fx, compute_gy
from fake_97_model import Fake97Model
from mcts_refinement_rl_prior import MCTSRefinement, ScoreGuidedPriorPolicy


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

    # 3. 生成 synthetic 初始解
    fake_model = Fake97Model(
        target_quality=0.88,
        quality_jitter=0.03,
        random_state=seed,
    )

    fx_init = fake_model.predict(
        fx_true,
        x=x,
    )

    # 4. policy prior / RL prior
    # ------------------------------------------------------------
    # 这里先用一个不依赖 fx_true 的 score-guided baseline policy。
    # 它会生成多个候选 action，估计哪些 action 更可能降低 score，
    # 然后给这些 action 写入不同的 Action.prior。
    #
    # 后续如果你训练好了 RL 模型，可以把这里替换成 LearnedPolicyPrior(model)。
    # ------------------------------------------------------------
    policy = ScoreGuidedPriorPolicy(
        num_candidates=18,
        temperature=0.20,
        min_prior=0.05,
        max_prior=8.0,
    )

    # 5. MCTS refinement
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

        # 新增：用 policy / RL 给 prior 不同值，引导 PUCT search
        policy_fn=policy,
        policy_prior_weight=1.0,
        prior_floor=0.05,
        prior_ceiling=8.0,

        random_state=seed + 100,
    )

    fx_refined, info = refiner.refine(
        fx_init,
        verbose=True,
        return_info=True,
    )

    # 6. 计算 forward 结果
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

    # 7. 打印指标
    print()
    print("========== Final Metrics ==========")
    print("fx relative mse before:", relative_mse(fx_init, fx_true))
    print("fx relative mse after :", relative_mse(fx_refined, fx_true))
    print("gy relative mse before:", relative_mse(gy_init, gy_true))
    print("gy relative mse after :", relative_mse(gy_refined, gy_true))
    print("MCTS best score       :", info["best_score"])
    print("root prior stats      :", info["root_prior_stats"])

    # 8. 可视化 fx
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
        label="Synthetic Initial Guess",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        x,
        fx_refined,
        label="Policy-Prior MCTS Refined",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Spectrum Refinement")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 9. 可视化 gy
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

    # 10. 可视化 MCTS score
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

    # 11. 可视化 root children 的 prior 变化范围
    plt.figure(figsize=(10, 5))

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

    plt.xlabel("Iteration")
    plt.ylabel("Prior")
    plt.title("Policy Prior Statistics")
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
