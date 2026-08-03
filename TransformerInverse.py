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


class InverseTransformerPeakResidual1D(nn.Module):
    """面向窄峰恢复的结构化 Transformer。

    与 V2 直接输出完整谱不同，本模型显式拆成两个分支：

      f_pred(s) = background_pred(s) + resonance_pred(s)

    background_pred 使用物理上正确的两维背景基函数；resonance_pred 保留
    自由曲线输出，但通过 Softplus 强制为非负。这样训练时不会再让大量背景点
    淹没局部峰，也不会像直接回归五个参数那样强制单一参数解。
    """

    def __init__(
        self,
        input_length=100,
        output_length=256,
        d_model=96,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        x_min=0.1764,
        x_max=6.0,
        y_min=-100.0,
        y_max=-6.0,
        normalize_coordinates=True,
        rms_normalize_io=True,
        rms_eps=1e-8,
        data_scale=160000.0,
        shift=400.0,
        peak_softplus_beta=2.0,
        peak_initial_bias=-2.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if output_length < 2:
            raise ValueError("output_length must be >= 2")

        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.normalize_coordinates = bool(normalize_coordinates)
        self.rms_normalize_io = bool(rms_normalize_io)
        self.rms_eps = float(rms_eps)
        self.data_scale = float(data_scale)
        self.shift = float(shift)
        self.peak_softplus_beta = float(peak_softplus_beta)

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

        # 只预测两个背景系数，再通过正确物理基函数生成背景曲线。
        self.background_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        # 局部卷积使 decoder 特征能够形成更尖锐的峰，而不是只输出平滑曲线。
        self.peak_feature_refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.peak_output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.peak_location_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        nn.init.constant_(self.peak_output_head[-1].bias, float(peak_initial_bias))

        if self.normalize_coordinates:
            y_grid = torch.linspace(-1.0, 1.0, self.input_length)
            x_grid = torch.linspace(-1.0, 1.0, self.output_length)
        else:
            y_grid = torch.linspace(float(y_min), float(y_max), self.input_length)
            x_grid = torch.linspace(float(x_min), float(x_max), self.output_length)

        x_physical = torch.linspace(float(x_min), float(x_max), self.output_length)
        basis_a2 = self.data_scale * x_physical / (x_physical + self.shift).square()
        basis_a3 = self.data_scale / (x_physical + self.shift).square()

        self.register_buffer("y_grid", y_grid.view(1, self.input_length, 1))
        self.register_buffer("x_grid", x_grid.view(1, self.output_length, 1))
        self.register_buffer("background_basis_a2", basis_a2.view(1, 1, -1))
        self.register_buffer("background_basis_a3", basis_a3.view(1, 1, -1))

    def _encode_decode(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError("Expected gy shape [B, 1, Ny], got %s" % (tuple(gy.shape),))
        if gy.shape[2] != self.input_length:
            raise ValueError(
                "Expected input length %d, got %d" % (self.input_length, gy.shape[2])
            )

        batch = gy.shape[0]
        io_scale = None
        gy_network = gy
        if self.rms_normalize_io:
            io_scale = torch.sqrt(
                torch.mean(gy.square(), dim=2, keepdim=True).clamp_min(self.rms_eps**2)
            )
            gy_network = gy / io_scale

        gy_tokens = gy_network.transpose(1, 2)
        y_pos = self.y_grid.expand(batch, -1, -1)
        x_pos = self.x_grid.expand(batch, -1, -1)
        src = self.src_norm(self.g_value_embed(gy_tokens) + self.y_pos_embed(y_pos))
        tgt = self.tgt_norm(self.x_query_embed(x_pos))
        memory = self.transformer.encoder(src)
        decoded = self.transformer.decoder(tgt, memory)
        return memory, decoded, io_scale

    def forward_with_components(self, gy):
        memory, decoded, io_scale = self._encode_decode(gy)

        pooled = torch.mean(memory, dim=1)
        background_coeff = self.background_head(pooled)
        a2_like = background_coeff[:, 0].view(-1, 1, 1)
        a3_like = background_coeff[:, 1].view(-1, 1, 1)
        background = (
            a2_like * self.background_basis_a2
            + a3_like * self.background_basis_a3
        )

        peak_features = decoded.transpose(1, 2)
        peak_features = peak_features + self.peak_feature_refine(peak_features)
        peak_tokens = peak_features.transpose(1, 2)
        peak_raw = self.peak_output_head(peak_tokens).transpose(1, 2)
        peak_location_logits = self.peak_location_head(peak_tokens).transpose(1, 2)
        peak_location_probability = torch.softmax(peak_location_logits, dim=2)
        location_envelope = (
            0.25 + 0.75 * self.output_length * peak_location_probability
        )
        resonance = (
            torch.nn.functional.softplus(peak_raw, beta=self.peak_softplus_beta)
            * location_envelope
        )

        if io_scale is not None:
            background = background * io_scale
            resonance = resonance * io_scale

        total = background + resonance
        return total, background, resonance, peak_location_logits

    def forward(self, gy):
        total, _, _, _ = self.forward_with_components(gy)
        return total
