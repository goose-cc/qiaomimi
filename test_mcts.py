import numpy as np
import matplotlib.pyplot as plt

from data_generate import generate_fx, compute_gy
from fake_97_model import Fake97Model
from mcts_refinement import MCTSRefinement


def mse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.mean((a - b) ** 2)


def relative_mse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.mean((a - b) ** 2) / (
        np.mean(b ** 2) + 1e-12
    )


def main():
    x = np.linspace(0.0, 2.0, 256)
    y = np.linspace(2.1, 10.0, 256)

    fx_true = generate_fx(x)
    gy_true = compute_gy(
        x,
        fx_true,
        y,
    )

    fake_model = Fake97Model(
        target_quality=0.97,
        quality_jitter=0.004,
        random_state=1,
    )

    fx_init = fake_model.predict(
        fx_true,
        x=x,
    )

    refiner = MCTSRefinement(
        x=x,
        y=y,
        gy_target=gy_true,
        iterations=500,
        rollout_depth=4,
        lambda_tv=0.03,
        lambda_curv=0.01,
        exploration=1.5,
        random_state=2,
    )

    fx_refined, info = refiner.refine(
        fx_init,
        verbose=True,
        return_info=True,
    )

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

    print()
    print("========== Metrics ==========")
    print("fx relative mse before:", relative_mse(fx_init, fx_true))
    print("fx relative mse after :", relative_mse(fx_refined, fx_true))
    print("gy relative mse before:", relative_mse(gy_init, gy_true))
    print("gy relative mse after :", relative_mse(gy_refined, gy_true))
    print("best MCTS score       :", info["best_score"])

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
    plt.title("MCTS Refinement Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        info["history"]["best_score"],
        label="Best Score",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("MCTS Score History")
    plt.legend()
    plt.tight_layout()
    plt.show()

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


if __name__ == "__main__":
    main()
