from __future__ import annotations

import torch

from mc_inverse_loss import MonteCarloInverseLoss


def make_loss(profile: str) -> MonteCarloInverseLoss:
    return MonteCarloInverseLoss(
        s_min=0.1764,
        s_max=6.0,
        q2_min=-100.0,
        q2_max=-6.0,
        output_points=100,
        input_points=100,
        profile=profile,
        normalization="relative",
        lambda_grad=0.1,
        lambda_physics=0.1,
        lambda_smooth=0.01,
        lambda_tv=0.01,
        lambda_tikhonov=0.01,
    )


def main() -> None:
    torch.manual_seed(20260802)
    f_true = torch.rand(8, 1, 100)
    params = torch.tensor(
        [[0.18, 0.01, 0.02, 1.0 + 0.3 * i, 0.10] for i in range(8)],
        dtype=torch.float32,
    )

    reference_loss = make_loss("pinn")
    g_clean = reference_loss.physics_forward_integral(f_true)
    if not torch.all(g_clean > 0):
        raise RuntimeError("positive f should produce positive g for s-q2 > 0")

    for profile in sorted(MonteCarloInverseLoss.VALID_PROFILES):
        loss_fn = make_loss(profile)
        kwargs = {"params": params} if profile == "peak_finetune" else {}
        perfect, perfect_logs = loss_fn(f_true.clone(), f_true, g_clean, **kwargs)
        for key in ("data_mse", "grad", "physics"):
            if perfect_logs[key] > 1e-10:
                raise RuntimeError(
                    f"{profile}: perfect prediction should have zero {key}, "
                    f"got {perfect_logs[key]}"
                )
        if not torch.isfinite(perfect):
            raise RuntimeError(f"{profile}: non-finite perfect-prediction loss")

        prediction = (f_true + 0.05 * torch.randn_like(f_true)).requires_grad_(True)
        loss, logs = loss_fn(prediction, f_true, g_clean, **kwargs)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{profile}: non-finite loss")
        loss.backward()
        if prediction.grad is None or not torch.isfinite(prediction.grad).all():
            raise RuntimeError(f"{profile}: invalid gradient")
        print(f"{profile:16s} loss={float(loss.detach()):.6e} logs={logs}")

    print("all loss profiles passed")


if __name__ == "__main__":
    main()
