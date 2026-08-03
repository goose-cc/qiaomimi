from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from TransformerInverse import InverseTransformer1D
from train_mc_parameter_pool_transformer_loss import OnlinePhysics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查蒙特卡洛逆问题中 g 信号与 Transformer 位置嵌入的尺度。"
    )
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = torch.device(args.device)
    path = Path(args.pool_dir) / "parameters.dat"
    rows = path.stat().st_size // (5 * 4)
    pool = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, 5))
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(rows, size=min(args.num_samples, rows), replace=False)
    params = torch.from_numpy(np.asarray(pool[ids], dtype=np.float32)).to(device)

    cfg = SimpleNamespace(
        physics_dtype="float32", noise_level=0.09, data_scale=160000.0,
        shift=400.0, s_min=0.1764, s_max=6.0, q2_min=-100.0, q2_max=-6.0,
        output_points=100, input_points=100, integration_points=128,
    )
    physics = OnlinePhysics(cfg, device)
    f, g = physics.make_clean_batch(params)

    raw = InverseTransformer1D(
        input_length=100, output_length=100, d_model=64, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=128,
        dropout=0.0, x_min=0.1764, x_max=6.0, y_min=-100.0, y_max=-6.0,
        normalize_coordinates=False, rms_normalize_io=False,
    ).to(device).eval()

    with torch.no_grad():
        tokens = g.unsqueeze(-1)
        value_signal = raw.g_value_embed(tokens) - raw.g_value_embed(torch.zeros_like(tokens))
        pos = raw.y_pos_embed(raw.y_grid.expand(len(g), -1, -1))
        value_rms = torch.sqrt(torch.mean(value_signal.square())).item()
        pos_rms = torch.sqrt(torch.mean(pos.square())).item()
        g_rms = torch.sqrt(torch.mean(g.square(), dim=1))
        f_rms = torch.sqrt(torch.mean(f.square(), dim=1))

    print("=" * 72)
    print(f"samples               : {len(g):,}")
    print(f"g RMS median          : {torch.median(g_rms).item():.6g}")
    print(f"g RMS p01/p99         : {torch.quantile(g_rms, 0.01).item():.6g} / {torch.quantile(g_rms, 0.99).item():.6g}")
    print(f"f RMS median          : {torch.median(f_rms).item():.6g}")
    print(f"g value signal embed  : {value_rms:.6g}")
    print(f"raw q2 position embed : {pos_rms:.6g}")
    print(f"position/value ratio  : {pos_rms / max(value_rms, 1e-30):.3f}x")
    print("=" * 72)
    if pos_rms > 20 * value_rms:
        print("诊断：位置嵌入显著大于 g 的有效信号，模型很可能忽略输入并输出平均曲线。")
        print("修复：启用坐标 [-1,1] 归一化和每样本 RMS I/O 归一化后重新训练。")
    else:
        print("尺度比没有明显异常；还需检查优化器、数据可辨识性与模型容量。")


if __name__ == "__main__":
    main()
