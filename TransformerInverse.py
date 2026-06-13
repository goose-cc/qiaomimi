import torch
import torch.nn as nn


class InverseTransformer1D(nn.Module):
    """用 Transformer 学习反算子：g(y) -> f(x)。

    输入 gy: [B, 1, Ny]
    输出 fx: [B, 1, Nx]

    注意：这里最后不加 ReLU，因为新数据中的 f(x) 可能出现负值。
    """

    def __init__(
        self,
        input_length=100,
        output_length=100,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        x_min=0.0,
        x_max=2.0,
        y_min=3.0,
        y_max=8.0,
    ):
        super().__init__()

        self.input_length = input_length
        self.output_length = output_length

        self.g_value_embed = nn.Linear(1, d_model)
        self.y_pos_embed = nn.Linear(1, d_model)
        self.x_query_embed = nn.Linear(1, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        y_grid = torch.linspace(y_min, y_max, input_length).view(1, input_length, 1)
        x_grid = torch.linspace(x_min, x_max, output_length).view(1, output_length, 1)
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

    def forward(self, gy):
        if gy.dim() != 3:
            raise ValueError(f'Expected gy shape [B, 1, Ny], but got {gy.shape}')

        B = gy.shape[0]
        gy = gy.transpose(1, 2)  # [B, Ny, 1]

        y_pos = self.y_grid.expand(B, -1, -1)
        x_pos = self.x_grid.expand(B, -1, -1)

        src = self.g_value_embed(gy) + self.y_pos_embed(y_pos)
        tgt = self.x_query_embed(x_pos)

        out = self.transformer(src=src, tgt=tgt)
        fx = self.output_head(out).transpose(1, 2)  # [B, 1, Nx]
        return fx
