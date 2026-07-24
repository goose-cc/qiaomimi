import numpy as np

data = np.load("./percent10/test.npz")

targets = [
    (0.4, 0.5),
    (0.6, 0.5),
    (1.0, 0.5),
    (1.2, 0.5),
    (1.4, 0.5),
    (0.8, 0.3),
    (0.8, 0.7),
]

for target_m, target_gamma in targets:
    distance = (
        (data["m"] - target_m) ** 2
        + (data["gamma"] - target_gamma) ** 2
    )

    index = int(np.argmin(distance))

    print(
        f"target=({target_m:.2f}, {target_gamma:.2f}) | "
        f"index={index} | "
        f"a1={data['a1'][index]:.6f} | "
        f"a2={data['a2'][index]:.6f} | "
        f"m={data['m'][index]:.6f} | "
        f"gamma={data['gamma'][index]:.6f} | "
        f"varying_param={data['varying_param'][index]}"
    )
