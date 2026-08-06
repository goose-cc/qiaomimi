#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experiment 2 quick checks: 1000-point output and grid-invariant difference loss."""

import torch

from TransformerInverse import InverseTransformer1D
from mc_inverse_loss import MonteCarloInverseLoss


def make_loss(output_points: int) -> MonteCarloInverseLoss:
    return MonteCarloInverseLoss(
        s_min=0.1764,
        s_max=6.0,
        q2_min=-100.0,
        q2_max=-6.0,
        output_points=output_points,
        input_points=100,
        profile="pinn",
        normalization="relative",
        lambda_grad=0.1,
        gradient_reference_points=100,
        lambda_physics=0.1,
    )


def main() -> None:
    torch.manual_seed(7)

    # 1) 原 Transformer 类无需改结构即可输出1000点。
    model = InverseTransformer1D(
        input_length=100,
        output_length=1000,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        x_min=0.1764,
        x_max=6.0,
        y_min=-100.0,
        y_max=-6.0,
        normalize_coordinates=True,
        rms_normalize_io=True,
    )
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 1, 100))
    assert tuple(y.shape) == (1, 1, 1000), tuple(y.shape)
    assert torch.isfinite(y).all()

    # 2) 线性函数在100点与1000点上的“参考网格差分”应接近一致。
    loss100 = make_loss(100)
    loss1000 = make_loss(1000)
    s100 = torch.linspace(0.1764, 6.0, 100).view(1, 1, -1)
    s1000 = torch.linspace(0.1764, 6.0, 1000).view(1, 1, -1)
    d100 = loss100.scaled_first_diff(2.0 * s100 + 1.0).mean()
    d1000 = loss1000.scaled_first_diff(2.0 * s1000 + 1.0).mean()
    assert torch.allclose(d100, d1000, rtol=2e-5, atol=2e-6), (d100, d1000)

    # 3) 1000点物理前向矩阵与损失可正常计算。
    f_true = torch.exp(-((s1000 - 2.0) / 0.2) ** 2)
    f_pred = f_true * 0.95
    g_ref = loss1000.physics_forward_integral(f_true)
    total, logs = loss1000(f_pred, f_true, g_ref)
    assert torch.isfinite(total)
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())

    print("EXP2 OUTPUT1000 TEST PASSED")
    print("model output shape:", tuple(y.shape))
    print("scaled linear diff (100): ", float(d100))
    print("scaled linear diff (1000):", float(d1000))
    print("1000-point test loss:", float(total))


if __name__ == "__main__":
    main()
