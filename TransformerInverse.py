import torch
import torch.nn as nn


class InverseTransformer1D(nn.Module):
    """用 Transformer 学习反算子：g(y) -> f(x)。

    输入 gy: [B, 1, Ny]
    输出 fx: [B, 1, Nx]

    对蒙特卡洛参数池问题，强烈建议同时启用：
      1. normalize_coordinates=True：把物理坐标映射到 [-1, 1] 后再做位置嵌入；
      2. rms_normalize_io=True：每条样本先除以输入 RMS，网络输出后再乘回同一 RMS。

    第二项利用了正向算子的线性齐次性：若 g 放大 c 倍，对应的 f 也放大 c 倍。
    这样可以避免小幅值 g 被位置嵌入和线性层 bias 淹没。
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
        normalize_coordinates=False,
        rms_normalize_io=False,
        rms_eps=1e-8,
    ):
        super().__init__()

        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.normalize_coordinates = bool(normalize_coordinates)
        self.rms_normalize_io = bool(rms_normalize_io)
        self.rms_eps = float(rms_eps)
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")

        self.g_value_embed = nn.Linear(1, d_model)
        self.y_pos_embed = nn.Linear(1, d_model)
        self.x_query_embed = nn.Linear(1, d_model)
        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)

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

        if self.normalize_coordinates:
            # Transformer 只需要知道相对位置；不应让 -100 这样的大坐标
            # 直接压过幅度仅约 1e-2 的观测信号。
            y_grid = torch.linspace(-1.0, 1.0, self.input_length)
            x_grid = torch.linspace(-1.0, 1.0, self.output_length)
        else:
            y_grid = torch.linspace(float(y_min), float(y_max), self.input_length)
            x_grid = torch.linspace(float(x_min), float(x_max), self.output_length)

        self.register_buffer("y_grid", y_grid.view(1, self.input_length, 1))
        self.register_buffer("x_grid", x_grid.view(1, self.output_length, 1))

    def forward(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError(f"Expected gy shape [B, 1, Ny], but got {gy.shape}")
        if gy.shape[2] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, but got {gy.shape[2]}"
            )

        batch = gy.shape[0]
        io_scale = None
        if self.rms_normalize_io:
            io_scale = torch.sqrt(
                torch.mean(gy.square(), dim=2, keepdim=True).clamp_min(self.rms_eps**2)
            )
            gy = gy / io_scale

        gy_tokens = gy.transpose(1, 2)  # [B, Ny, 1]
        y_pos = self.y_grid.expand(batch, -1, -1)
        x_pos = self.x_grid.expand(batch, -1, -1)

        src = self.src_norm(
            self.g_value_embed(gy_tokens) + self.y_pos_embed(y_pos)
        )
        tgt = self.tgt_norm(self.x_query_embed(x_pos))

        out = self.transformer(src=src, tgt=tgt)
        fx = self.output_head(out).transpose(1, 2)  # [B, 1, Nx]

        if io_scale is not None:
            fx = fx * io_scale
        return fx

class InverseBiLSTMTransformer1D(InverseTransformer1D):
    """在原 V2 Transformer 输入端叠加双向 LSTM。

    数据流：
        g(q^2) -> 数值/位置嵌入 -> BiLSTM -> 原 Transformer -> f(s)

    BiLSTM 只处理 100 个输入观测点，不对 1000 个输出点做循环生成，
    因而保持原 Transformer 解码方式，并把结构变化限制在输入特征提取端。
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
        normalize_coordinates=False,
        rms_normalize_io=False,
        rms_eps=1e-8,
        lstm_hidden_size=32,
        lstm_num_layers=2,
        lstm_dropout=0.1,
        lstm_residual=True,
    ):
        super().__init__(
            input_length=input_length,
            output_length=output_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            normalize_coordinates=normalize_coordinates,
            rms_normalize_io=rms_normalize_io,
            rms_eps=rms_eps,
        )

        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_dropout = float(lstm_dropout)
        self.lstm_residual = bool(lstm_residual)

        if self.lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        if self.lstm_num_layers <= 0:
            raise ValueError("lstm_num_layers must be positive")
        if not (0.0 <= self.lstm_dropout < 1.0):
            raise ValueError("lstm_dropout must be in [0, 1)")

        effective_dropout = self.lstm_dropout if self.lstm_num_layers > 1 else 0.0
        self.input_bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=effective_dropout,
            bidirectional=True,
            batch_first=True,
        )

        lstm_output_size = 2 * self.lstm_hidden_size
        if lstm_output_size == d_model:
            self.lstm_projection = nn.Identity()
        else:
            self.lstm_projection = nn.Linear(lstm_output_size, d_model)
        self.lstm_output_norm = nn.LayerNorm(d_model)

        # 让 LSTM 初始时更容易保留较长范围的信息。
        for name, parameter in self.input_bilstm.named_parameters():
            if "bias" in name:
                gate_size = parameter.numel() // 4
                with torch.no_grad():
                    parameter[gate_size : 2 * gate_size].fill_(1.0)

    def forward(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError(f"Expected gy shape [B, 1, Ny], but got {gy.shape}")
        if gy.shape[2] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, but got {gy.shape[2]}"
            )

        batch = gy.shape[0]
        io_scale = None
        if self.rms_normalize_io:
            io_scale = torch.sqrt(
                torch.mean(gy.square(), dim=2, keepdim=True).clamp_min(self.rms_eps**2)
            )
            gy = gy / io_scale

        gy_tokens = gy.transpose(1, 2)  # [B, Ny, 1]
        y_pos = self.y_grid.expand(batch, -1, -1)
        x_pos = self.x_grid.expand(batch, -1, -1)

        embedded_src = self.src_norm(
            self.g_value_embed(gy_tokens) + self.y_pos_embed(y_pos)
        )
        lstm_features, _ = self.input_bilstm(embedded_src)
        lstm_features = self.lstm_projection(lstm_features)
        if self.lstm_residual:
            src = self.lstm_output_norm(embedded_src + lstm_features)
        else:
            src = self.lstm_output_norm(lstm_features)

        tgt = self.tgt_norm(self.x_query_embed(x_pos))
        out = self.transformer(src=src, tgt=tgt)
        fx = self.output_head(out).transpose(1, 2)  # [B, 1, Nx]

        if io_scale is not None:
            fx = fx * io_scale
        return fx

