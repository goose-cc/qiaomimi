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

class ParametricInverseTransformer1D(nn.Module):
    """直接从 g(q^2) 预测 5 个物理参数，再由解析物理公式重建 f(s)。

    这是可辨识性诊断模型，不是为了继续堆网络：
      g(q^2), 100 points -> Transformer encoder -> [a1,a2,a3,m,gamma]

    为了既稳定归一化输入、又不丢失样本整体幅度信息：
      1. 每条 g 除以自身 RMS；
      2. log(RMS) 作为额外全局特征送入参数 head。
    因此输入尺度信息仍然保留。
    """

    def __init__(
        self,
        input_length=100,
        output_length=1000,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        y_min=-100.0,
        y_max=-6.0,
        x_min=0.1764,
        x_max=6.0,
        shift=400.0,
        data_scale=160000.0,
        rms_eps=1e-8,
    ):
        super().__init__()
        from mc_parametric import parameter_bounds

        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.shift = float(shift)
        self.data_scale = float(data_scale)
        self.rms_eps = float(rms_eps)
        if self.input_length <= 1 or self.output_length <= 1:
            raise ValueError('input_length and output_length must be > 1')
        if self.rms_eps <= 0:
            raise ValueError('rms_eps must be positive')
        if d_model % nhead != 0:
            raise ValueError('d_model must be divisible by nhead')

        self.g_value_embed = nn.Linear(1, d_model)
        self.y_pos_embed = nn.Linear(1, d_model)
        self.src_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # +1 is log(RMS), so normalization does not destroy absolute-amplitude information.
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model + 1, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 5),
        )

        y_grid = torch.linspace(-1.0, 1.0, self.input_length)
        s_grid = torch.linspace(float(x_min), float(x_max), self.output_length)
        lower, upper = parameter_bounds()
        self.register_buffer('y_grid', y_grid.view(1, self.input_length, 1))
        self.register_buffer('s_grid', s_grid)
        self.register_buffer('parameter_lower', lower)
        self.register_buffer('parameter_upper', upper)

    def predict_parameters(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError(f'Expected gy shape [B, 1, Ny], but got {tuple(gy.shape)}')
        if gy.shape[2] != self.input_length:
            raise ValueError(
                f'Expected input length {self.input_length}, but got {gy.shape[2]}'
            )

        batch = gy.shape[0]
        rms = torch.sqrt(
            torch.mean(gy.square(), dim=2, keepdim=True).clamp_min(self.rms_eps**2)
        )
        gy_norm = gy / rms
        tokens = gy_norm.transpose(1, 2)
        pos = self.y_grid.expand(batch, -1, -1).to(dtype=tokens.dtype)
        src = self.src_norm(self.g_value_embed(tokens) + self.y_pos_embed(pos))
        encoded = self.encoder_norm(self.encoder(src))
        pooled = encoded.mean(dim=1)

        # Clamp only for numerical safety; RMS itself remains fully represented.
        log_rms = torch.log(rms[:, 0, 0].clamp_min(self.rms_eps)).unsqueeze(1)
        logits = self.parameter_head(torch.cat([pooled, log_rms], dim=1))
        unit = torch.sigmoid(logits)
        lower = self.parameter_lower.to(dtype=unit.dtype, device=unit.device)
        upper = self.parameter_upper.to(dtype=unit.dtype, device=unit.device)
        return lower + unit * (upper - lower)

    def forward_with_parameters(self, gy):
        from mc_parametric import scaled_spectral_components

        params = self.predict_parameters(gy)
        total, _, _ = scaled_spectral_components(
            params,
            self.s_grid,
            shift=self.shift,
            data_scale=self.data_scale,
        )
        return total.unsqueeze(1), params

    def forward(self, gy):
        prediction, _ = self.forward_with_parameters(gy)
        return prediction

class PeakParametricInverseTransformer1D(nn.Module):
    """Exp8 strong model: shared Transformer encoder + separate physics heads.

    Output remains the physical parameter order [a1, a2, a3, m, gamma], but
    gamma is represented internally on a logarithmic scale.  The heads follow
    the actual rho model:
      resonance head -> [a1, m, log(gamma)]
      background head -> [a2, a3]

    This keeps the analytic physics decoder and gives the weak resonance
    variables their own capacity instead of forcing all five parameters through
    one undifferentiated linear head.
    """

    def __init__(
        self,
        input_length=100,
        output_length=1000,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        y_min=-100.0,
        y_max=-6.0,
        x_min=0.1764,
        x_max=6.0,
        shift=400.0,
        data_scale=160000.0,
        rms_eps=1e-8,
        gamma_log_floor=1e-5,
    ):
        super().__init__()
        import math
        from mc_parametric import parameter_bounds

        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.shift = float(shift)
        self.data_scale = float(data_scale)
        self.rms_eps = float(rms_eps)
        self.gamma_log_floor = float(gamma_log_floor)
        if self.input_length <= 1 or self.output_length <= 1:
            raise ValueError('input_length and output_length must be > 1')
        if self.rms_eps <= 0 or self.gamma_log_floor <= 0:
            raise ValueError('rms_eps and gamma_log_floor must be positive')
        if d_model % nhead != 0:
            raise ValueError('d_model must be divisible by nhead')

        self.g_value_embed = nn.Linear(1, d_model)
        self.y_pos_embed = nn.Linear(1, d_model)
        self.src_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        head_in = d_model + 1
        self.background_head = nn.Sequential(
            nn.Linear(head_in, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 2),   # a2, a3
        )
        self.resonance_head = nn.Sequential(
            nn.Linear(head_in, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3),   # a1, m, log-gamma unit coordinate
        )

        y_grid = torch.linspace(-1.0, 1.0, self.input_length)
        s_grid = torch.linspace(float(x_min), float(x_max), self.output_length)
        lower, upper = parameter_bounds(open_bound_eps=self.gamma_log_floor)
        if not (0 < self.gamma_log_floor < float(upper[4])):
            raise ValueError('gamma_log_floor must be smaller than gamma_max')
        self.register_buffer('y_grid', y_grid.view(1, self.input_length, 1))
        self.register_buffer('s_grid', s_grid)
        self.register_buffer('parameter_lower', lower)
        self.register_buffer('parameter_upper', upper)
        self.register_buffer('log_gamma_lower', torch.tensor(math.log(self.gamma_log_floor), dtype=torch.float32))
        self.register_buffer('log_gamma_upper', torch.tensor(math.log(float(upper[4])), dtype=torch.float32))

        # Start gamma near 0.25 instead of the geometric midpoint of the log
        # range (which would be excessively narrow at initialization).
        gamma_init = min(max(0.25, self.gamma_log_floor * 1.01), float(upper[4]) * 0.99)
        unit_init = (math.log(gamma_init) - math.log(self.gamma_log_floor)) / (
            math.log(float(upper[4])) - math.log(self.gamma_log_floor)
        )
        unit_init = min(max(unit_init, 1e-4), 1.0 - 1e-4)
        bias = math.log(unit_init / (1.0 - unit_init))
        with torch.no_grad():
            self.resonance_head[-1].bias[2].fill_(bias)

    def _encode(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError(f'Expected gy shape [B, 1, Ny], but got {tuple(gy.shape)}')
        if gy.shape[2] != self.input_length:
            raise ValueError(f'Expected input length {self.input_length}, but got {gy.shape[2]}')
        batch = gy.shape[0]
        rms = torch.sqrt(torch.mean(gy.square(), dim=2, keepdim=True).clamp_min(self.rms_eps**2))
        gy_norm = gy / rms
        tokens = gy_norm.transpose(1, 2)
        pos = self.y_grid.expand(batch, -1, -1).to(dtype=tokens.dtype)
        src = self.src_norm(self.g_value_embed(tokens) + self.y_pos_embed(pos))
        encoded = self.encoder_norm(self.encoder(src))
        pooled = encoded.mean(dim=1)
        log_rms = torch.log(rms[:, 0, 0].clamp_min(self.rms_eps)).unsqueeze(1)
        return torch.cat([pooled, log_rms], dim=1)

    def predict_parameters(self, gy):
        feat = self._encode(gy)
        bg_unit = torch.sigmoid(self.background_head(feat))
        res_unit = torch.sigmoid(self.resonance_head(feat))

        lower = self.parameter_lower.to(dtype=feat.dtype, device=feat.device)
        upper = self.parameter_upper.to(dtype=feat.dtype, device=feat.device)

        a2 = lower[1] + bg_unit[:, 0] * (upper[1] - lower[1])
        a3 = lower[2] + bg_unit[:, 1] * (upper[2] - lower[2])
        a1 = lower[0] + res_unit[:, 0] * (upper[0] - lower[0])
        mass = lower[3] + res_unit[:, 1] * (upper[3] - lower[3])

        log_lo = self.log_gamma_lower.to(dtype=feat.dtype, device=feat.device)
        log_hi = self.log_gamma_upper.to(dtype=feat.dtype, device=feat.device)
        log_gamma = log_lo + res_unit[:, 2] * (log_hi - log_lo)
        gamma = torch.exp(log_gamma)
        return torch.stack([a1, a2, a3, mass, gamma], dim=1)

    def forward_with_parameters(self, gy):
        from mc_parametric import scaled_spectral_components
        params = self.predict_parameters(gy)
        total, _, _ = scaled_spectral_components(
            params, self.s_grid, shift=self.shift, data_scale=self.data_scale
        )
        return total.unsqueeze(1), params

    def forward(self, gy):
        prediction, _ = self.forward_with_parameters(gy)
        return prediction

