import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakInversionTransformer(nn.Module):
    """
    Transformer-based inverse model.

    input:  g(y), shape [B, 1, num_y_points]
    output: f(x), shape [B, 1, num_x_points]
    """

    def __init__(
        self,
        input_length=100,
        output_length=100,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.input_length = input_length
        self.output_length = output_length

        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, gy):
        # gy: [B, 1, L]
        z = gy.transpose(1, 2)          # [B, L, 1]
        z = self.input_proj(z)          # [B, L, d_model]
        z = z + self.pos_embedding[:, :z.size(1), :]

        z = self.encoder(z)

        out = self.output_head(z)       # [B, L, 1]
        out = out.transpose(1, 2)       # [B, 1, L]

        if out.size(-1) != self.output_length:
            out = F.interpolate(
                out,
                size=self.output_length,
                mode="linear",
                align_corners=False
            )

        return out
