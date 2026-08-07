from __future__ import annotations

import torch

from TransformerInverse import InverseBiLSTMTransformer1D


def main() -> None:
    torch.manual_seed(7)
    model = InverseBiLSTMTransformer1D(
        input_length=100,
        output_length=1000,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        normalize_coordinates=True,
        rms_normalize_io=True,
        lstm_hidden_size=8,
        lstm_num_layers=2,
        lstm_dropout=0.0,
        lstm_residual=True,
    )

    x = torch.randn(2, 1, 100)
    y = model(x)
    assert y.shape == (2, 1, 1000), y.shape
    assert torch.isfinite(y).all()

    loss = y.square().mean()
    loss.backward()
    lstm_grad = model.input_bilstm.weight_ih_l0.grad
    assert lstm_grad is not None
    assert torch.isfinite(lstm_grad).all()

    parameter_count = sum(p.numel() for p in model.parameters())
    print("EXP5 BILSTM+TRANSFORMER TEST PASSED")
    print("output shape:", tuple(y.shape))
    print("test model parameters:", parameter_count)


if __name__ == "__main__":
    main()
