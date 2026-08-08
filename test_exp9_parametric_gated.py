from types import SimpleNamespace

import torch

from TransformerInverse import PeakParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric_gated import (
    conditional_parameter_mse,
    gated_log_peak_height_mse,
    gated_log_width_mse,
    gated_relative_mse,
    gradient_relative_mse,
    relative_mse,
    resonance_visibility,
    smooth_visibility_gate,
    visibility_mse,
    weak_resonance_excess_mse,
)


def main():
    torch.manual_seed(9)
    device = torch.device("cpu")

    args = SimpleNamespace(
        physics_dtype="float32",
        noise_level=0.0,
        data_scale=160000.0,
        shift=400.0,
        s_min=0.1764,
        s_max=6.0,
        q2_min=-100.0,
        q2_max=-6.0,
        output_points=1000,
        input_points=100,
        integration_points=64,
    )
    physics = OnlinePhysics(args, device)

    # One clearly visible narrow resonance + one nearly background-like sample.
    true = torch.tensor(
        [
            [0.15, 0.02, -0.01, 1.20, 0.025],
            [0.002, 0.03, 0.02, 0.80, 0.60],
        ],
        dtype=torch.float32,
    )
    true_total, true_res, _ = physics.components(true)
    g = physics.forward_from_parameters(true)

    true_vis = resonance_visibility(true_res, true_total)
    gate = smooth_visibility_gate(true_vis, 0.05, 0.20)
    weak_gate = 1.0 - gate

    assert true_vis[0] > true_vis[1], (true_vis[0], true_vis[1])
    assert gate[0] > gate[1], (gate[0], gate[1])

    model = PeakParametricInverseTransformer1D(
        input_length=100,
        output_length=1000,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        gamma_log_floor=1e-5,
    )
    pred = model.predict_parameters(g.unsqueeze(1))
    pred_total, pred_res, _ = physics.components(pred)
    pred_g = physics.forward_from_parameters(pred)

    base = torch.tensor([1.0, 1.0, 1.0, 0.25, 0.25])
    boost = torch.tensor([0.5, 0.0, 0.0, 1.75, 2.25])

    l_param = conditional_parameter_mse(
        pred,
        true,
        model.parameter_lower,
        model.parameter_upper,
        base,
        boost,
        gate,
        gamma_floor=1e-5,
        boost_scale=1.0,
    )
    l_spec = relative_mse(pred_total, true_total, floor=true_total)
    l_res = gated_relative_mse(pred_res, true_res, gate, floor=true_total, floor_fraction=0.02)
    l_grad = gradient_relative_mse(pred_total, true_total, true_total)
    l_phys = relative_mse(pred_g, g)
    l_width = gated_log_width_mse(pred, true, gate)
    l_peak = gated_log_peak_height_mse(pred, true, gate, 400.0, 160000.0)
    l_vis, _, _ = visibility_mse(pred_res, pred_total, true_res, true_total)
    l_weak = weak_resonance_excess_mse(
        pred_res,
        true_res,
        true_total,
        weak_gate,
        tolerance_fraction=0.02,
    )

    loss = (
        l_param
        + l_spec
        + 0.10 * l_grad
        + 0.25 * l_phys
        + 1.5 * l_res
        + 1.0 * l_width
        + 0.5 * l_peak
        + 0.75 * l_vis
        + 1.0 * l_weak
    )
    loss.backward()

    grad = sum(
        float(p.grad.abs().sum())
        for p in model.parameters()
        if p.grad is not None
    )

    assert pred.shape == (2, 5)
    assert pred_total.shape == (2, 1000)
    assert torch.isfinite(loss)
    assert grad > 0
    assert torch.all(pred[:, 4] > 0)

    print("EXP9 CONDITIONAL GATED TEST PASSED")
    print("true visibility:", true_vis.tolist())
    print("gate:", gate.tolist())
    print("loss:", float(loss))
    print("model parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
