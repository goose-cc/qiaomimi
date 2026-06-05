import numpy as np
import matplotlib.pyplot as plt

from data_generate import generate_fx
from fake_97_model import Fake97Model


def main():
    x = np.linspace(0.0, 2.0, 256)

    fx_true = generate_fx(x)

    model = Fake97Model(
        target_quality=0.97,
        quality_jitter=0.004,
        random_state=0,
    )

    fx_pred = model.predict(
        fx_true,
        x=x,
    )

    rel_error = Fake97Model.relative_l2_error(
        fx_true,
        fx_pred,
    )

    quality = Fake97Model.quality(
        fx_true,
        fx_pred,
    )

    print("Relative L2 error:", rel_error)
    print("Synthetic quality :", quality)

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        fx_true,
        label="True Spectrum",
        linewidth=3,
    )

    plt.plot(
        x,
        fx_pred,
        label="Synthetic 97% Model Output",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Test Fake97Model")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
