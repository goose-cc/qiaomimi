from __future__ import annotations

from types import SimpleNamespace

import torch

from TransformerInverse import ParametricInverseTransformer1D
from mc_inverse_loss import MonteCarloInverseLoss
from mc_online_physics import OnlinePhysics
from mc_parametric import normalized_parameter_mse, relative_curve_mse


def main():
    torch.manual_seed(20260802)
    args = SimpleNamespace(
        physics_dtype="float32",
        noise_level=0.09,
        data_scale=160000.0,
        shift=400.0,
        s_min=0.1764,
        s_max=6.0,
        q2_min=-100.0,
        q2_max=-6.0,
        output_points=100,
        input_points=100,
        integration_points=64,
    )
    physics = OnlinePhysics(args, torch.device("cpu"))
    true_params = torch.tensor(
        [[0.15, 0.02, -0.01, 0.9, 0.2], [0.10, 0.01, 0.00, 1.5, 0.4]],
        dtype=torch.float32,
    )
    true_f, g_clean = physics.make_clean_batch(true_params)
    model = ParametricInverseTransformer1D(
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    predicted_f, predicted_params = model.forward_with_parameters(g_clean.unsqueeze(1))
    predicted_g = physics.forward_from_parameters(predicted_params)
    loss_fn = MonteCarloInverseLoss(
        s_min=args.s_min,
        s_max=args.s_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        output_points=100,
        input_points=100,
        profile="pinn",
        lambda_grad=0.05,
        lambda_physics=0.01,
    )
    curve_loss, _ = loss_fn(
        predicted_f,
        true_f.unsqueeze(1),
        g_clean.unsqueeze(1),
        g_pred_override=predicted_g.unsqueeze(1),
    )
    parameter_loss = normalized_parameter_mse(
        predicted_params,
        true_params,
        model.parameter_lower,
        model.parameter_upper,
        torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]),
    )
    _, true_resonance, _ = physics.components(true_params)
    _, predicted_resonance, _ = physics.components(predicted_params)
    resonance_loss = relative_curve_mse(
        predicted_resonance,
        true_resonance,
        scale_floor=true_f,
    )
    total = curve_loss + parameter_loss + resonance_loss
    total.backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not torch.isfinite(total) or gradient_sum <= 0:
        raise RuntimeError("parametric forward/backward failed")
    if predicted_f.shape != (2, 1, 100) or predicted_params.shape != (2, 5):
        raise RuntimeError("unexpected output shape")
    print("parametric model forward/backward passed")
    print("loss=%.6e gradient_sum=%.6e" % (float(total.detach()), gradient_sum))


if __name__ == "__main__":
    main()
