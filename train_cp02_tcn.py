from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import argparse, csv, json, math, random, time

import numpy as np
import torch
import torch.nn as nn


RECOVERY = {"a1_abs": 0.005, "m_abs": 0.03, "gamma_factor": 1.2}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def configure_backend(device: torch.device, deterministic: bool, tf32: bool) -> dict:
    info = {"deterministic": bool(deterministic), "cudnn_benchmark": False, "tf32": False}
    if device.type != "cuda":
        return info

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    allow_tf32 = bool(tf32 and not deterministic)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
        torch.set_float32_matmul_precision("high")
    info["tf32"] = allow_tf32
    return info


def amp_context(enabled: bool):
    if enabled:
        return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    return nullcontext()


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_norm(path: Path):
    z = np.load(path, allow_pickle=False)
    input_mean = np.asarray(z["input_mean"], np.float32)
    input_std = np.maximum(np.asarray(z["input_std"], np.float32), 1e-12)
    target_mean = np.asarray(z["target_mean"], np.float32)
    target_std = np.maximum(np.asarray(z["target_std"], np.float32), 1e-12)

    target_names = None
    if "target_names" in z.files:
        target_names = [str(x) for x in np.asarray(z["target_names"]).tolist()]
    z.close()

    if target_mean.shape != (3,) or target_std.shape != (3,):
        raise ValueError("target_mean/target_std must have shape (3,)")
    if target_names is not None:
        cleaned = [x.lower().replace(" ", "") for x in target_names]
        if cleaned != ["a1", "m", "log_gamma"]:
            raise ValueError(f"Unexpected target_names={target_names}")
    return input_mean, input_std, target_mean, target_std


def extract_q2(z, data_root: Path, length: int):
    q2 = None
    if "q2" in z.files:
        q2 = np.asarray(z["q2"], np.float64)
        if q2.ndim == 2:
            if q2.shape[0] == 1:
                q2 = q2[0]
            elif np.allclose(q2, q2[0][None, :], rtol=0, atol=1e-12):
                q2 = q2[0]
            else:
                raise ValueError("Per-sample q2 arrays differ.")
    elif (data_root / "q2.npy").exists():
        q2 = np.asarray(np.load(data_root / "q2.npy"), np.float64)

    if q2 is None:
        return None
    q2 = q2.reshape(-1)
    if len(q2) != length:
        raise ValueError(f"q2 length {len(q2)} != gy length {length}")
    return q2


class CP02Dataset:
    def __init__(self, path: Path, norm, data_root: Path):
        self.path = Path(path)
        input_mean, input_std, target_mean, target_std = norm
        z = np.load(self.path, allow_pickle=False)

        self.x_raw = np.asarray(z["gy"], np.float32)
        if self.x_raw.ndim != 2:
            raise ValueError(f"gy must be [N,L], got {self.x_raw.shape}")

        if all(k in z.files for k in ("a1", "m", "gamma")):
            a1 = np.asarray(z["a1"], np.float32)
            m = np.asarray(z["m"], np.float32)
            gamma = np.asarray(z["gamma"], np.float32)
        elif "params3" in z.files:
            p = np.asarray(z["params3"], np.float32)
            a1, m, gamma = p[:, 0], p[:, 1], p[:, 2]
        else:
            raise KeyError(f"{self.path}: need a1/m/gamma or params3")

        if np.any(gamma <= 0):
            raise ValueError("gamma must be > 0")

        self.y_raw = np.stack([a1, m, np.log(gamma)], axis=1).astype(np.float32)
        self.x = ((self.x_raw - input_mean) / input_std).astype(np.float32)
        self.y = ((self.y_raw - target_mean) / target_std).astype(np.float32)

        n = len(self.x)
        self.ids = (
            np.asarray(z["physical_state_id"]).astype(str)
            if "physical_state_id" in z.files
            else np.asarray([str(i) for i in range(n)])
        )
        self.margin = (
            np.asarray(z["continuous_margin"], np.float64)
            if "continuous_margin" in z.files
            else np.full(n, np.nan)
        )
        self.noise_level = (
            np.asarray(z["noise_level"], np.float64)
            if "noise_level" in z.files
            else np.full(n, np.nan)
        )
        self.q2 = extract_q2(z, data_root, self.x.shape[1])
        z.close()

        if len(self.y) != n:
            raise ValueError("gy/target length mismatch")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{self.path}: NaN/Inf found")

    def __len__(self):
        return len(self.x)


def make_coordinate_channels(q2, length: int):
    if q2 is None:
        idx = np.linspace(-1.0, 1.0, length, dtype=np.float32)
        return torch.from_numpy(idx[None, :]), "index_fallback"

    q2 = np.asarray(q2, np.float64)
    qmin, qmax = float(q2.min()), float(q2.max())
    qlin = 2.0 * (q2 - qmin) / max(qmax - qmin, 1e-30) - 1.0

    logabs = np.log(np.maximum(np.abs(q2), 1e-12))
    lmin, lmax = float(logabs.min()), float(logabs.max())
    logabs = 2.0 * (logabs - lmin) / max(lmax - lmin, 1e-30) - 1.0
    return torch.from_numpy(np.stack([qlin, logabs], axis=0).astype(np.float32)), "q2_linear_plus_logabs"


class ResidualTCNBlockNoNorm(nn.Module):
    """
    Amplitude-preserving residual block.

    Important: no GroupNorm/LayerNorm/InstanceNorm is applied to the activation.
    Per-sample normalization was suppressing absolute-amplitude information that
    is important for a1.
    """
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.03):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("Use an odd kernel size.")
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = self.skip(x)
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        return self.act(x + r)


class AmpPreserveTCN3P(nn.Module):
    """
    TCN that preserves both amplitude and q2 position.

    Changes from v2:
      1) removes GroupNorm;
      2) replaces global mean/max-only compression with positional pooling;
      3) adds a low-dimensional direct observation shortcut so exact global
         amplitude/position information cannot be erased by the convolution path.
    """
    def __init__(
        self,
        input_length: int,
        coord_channels: torch.Tensor,
        channels=(64, 96, 128, 160),
        kernel=5,
        dropout=0.03,
        pool_bins=12,
        shortcut_dim=128,
        use_shortcut=True,
    ):
        super().__init__()
        self.input_length = int(input_length)
        self.use_shortcut = bool(use_shortcut)

        self.register_buffer("coord", coord_channels[None, :, :].float(), persistent=True)

        blocks = []
        in_ch = 1 + coord_channels.shape[0]
        for i, out_ch in enumerate(channels):
            blocks.append(
                ResidualTCNBlockNoNorm(
                    in_ch, out_ch, kernel=kernel, dilation=2 ** i, dropout=dropout
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)

        self.avg_pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(pool_bins)
        tcn_dim = 2 * channels[-1] * pool_bins

        # Direct observation shortcut. It is intentionally small: it does not
        # replace the TCN; it carries exact whole-trace amplitude/position cues.
        if self.use_shortcut:
            self.obs_shortcut = nn.Sequential(
                nn.Linear(self.input_length, 192),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(192, shortcut_dim),
                nn.GELU(),
            )
            shortcut_out = shortcut_dim
        else:
            self.obs_shortcut = None
            shortcut_out = 0

        # Simple global amplitude descriptors of the normalized trace.
        stats_dim = 7  # mean, std, rms, min, max, first, last
        fusion_in = tcn_dim + shortcut_out + stats_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 160),
            nn.GELU(),
        )

        def head():
            return nn.Sequential(
                nn.Linear(160, 80),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(80, 1),
            )

        self.head_a1 = head()
        self.head_m = head()
        self.head_loggamma = head()

    @staticmethod
    def trace_stats(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-12)
        mn = x.amin(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        first = x[:, :1]
        last = x[:, -1:]
        return torch.cat([mean, std, rms, mn, mx, first, last], dim=1)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected [B,L], got {tuple(x.shape)}")
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length {x.shape[1]} != expected {self.input_length}")

        c = self.coord.expand(x.shape[0], -1, -1)
        h = torch.cat([x.unsqueeze(1), c], dim=1)
        h = self.backbone(h)

        # Preserve coarse q2 location instead of reducing the whole trace to
        # only one mean and one max per channel.
        h_avg = self.avg_pool(h).flatten(1)
        h_max = self.max_pool(h).flatten(1)

        parts = [h_avg, h_max, self.trace_stats(x)]
        if self.obs_shortcut is not None:
            parts.append(self.obs_shortcut(x))

        fused = self.fusion(torch.cat(parts, dim=1))
        return torch.cat(
            [self.head_a1(fused), self.head_m(fused), self.head_loggamma(fused)], dim=1
        )


def cuda_mem_gb() -> str:
    if not torch.cuda.is_available():
        return "n/a"
    a = torch.cuda.memory_allocated() / (1024 ** 3)
    r = torch.cuda.memory_reserved() / (1024 ** 3)
    return f"{a:.2f}/{r:.2f} GB alloc/reserved"


def normalized_to_physical(pred_norm, target_mean, target_std):
    raw = pred_norm * target_std[None, :] + target_mean[None, :]
    a1 = raw[:, 0]
    m = raw[:, 1]
    gamma = np.exp(np.clip(raw[:, 2], -50.0, 50.0))
    return a1, m, gamma


def recovery_from_pred_norm(pred_norm, ds, norm):
    _, _, target_mean, target_std = norm
    pa1, pm, pg = normalized_to_physical(pred_norm, target_mean, target_std)

    ta1 = ds.y_raw[:, 0].astype(np.float64)
    tm = ds.y_raw[:, 1].astype(np.float64)
    tg = np.exp(ds.y_raw[:, 2].astype(np.float64))

    e1 = np.abs(pa1 - ta1)
    em = np.abs(pm - tm)
    gf = np.maximum(
        pg / np.maximum(tg, 1e-300),
        tg / np.maximum(pg, 1e-300),
    )
    r1 = float(np.mean(e1 <= RECOVERY["a1_abs"]))
    rm = float(np.mean(em <= RECOVERY["m_abs"]))
    rg = float(np.mean(gf <= RECOVERY["gamma_factor"]))
    rj = float(np.mean(
        (e1 <= RECOVERY["a1_abs"]) &
        (em <= RECOVERY["m_abs"]) &
        (gf <= RECOVERY["gamma_factor"])
    ))
    return {"R_a1": r1, "R_m": rm, "R_gamma": rg, "R_joint": rj, "min_R": min(r1, rm, rg)}


def train_epoch(
    model, x, y, lossfn, optimizer, scaler, amp_enabled,
    batch_size, seed, epoch, log_every
):
    model.train()
    n = x.shape[0]
    nb = math.ceil(n / batch_size)
    perm = torch.randperm(n, device=x.device)
    total = 0.0
    seen = 0

    if x.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for bi in range(nb):
        s = bi * batch_size
        e = min(n, s + batch_size)
        idx = perm[s:e]
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled):
            pred = model(xb)
            loss = lossfn(pred, yb)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = e - s
        total += float(loss.detach().float().item()) * bs
        seen += bs

        if log_every > 0 and ((bi + 1) % log_every == 0 or bi + 1 == nb):
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            dt = max(time.perf_counter() - t0, 1e-9)
            print(
                f"seed={seed} ep={epoch} batch={bi+1}/{nb} "
                f"loss={total/max(seen,1):.6f} {seen/dt:.1f} samples/s "
                f"gpu_mem={cuda_mem_gb()}",
                flush=True,
            )

    if x.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)
    return total / seen, elapsed, seen / elapsed


@torch.no_grad()
def eval_loss_and_pred(model, x, y, lossfn, amp_enabled, batch_size):
    model.eval()
    losses = 0.0
    chunks = []
    n = x.shape[0]
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        with amp_context(amp_enabled):
            p = model(x[s:e])
            loss = lossfn(p, y[s:e])
        losses += float(loss.detach().float().item()) * (e - s)
        chunks.append(p.float().cpu())
    return losses / n, torch.cat(chunks, dim=0).numpy()


@torch.no_grad()
def predict(model, x, amp_enabled, batch_size):
    model.eval()
    chunks = []
    for s in range(0, x.shape[0], batch_size):
        with amp_context(amp_enabled):
            p = model(x[s:s+batch_size])
        chunks.append(p.float().cpu())
    return torch.cat(chunks, dim=0).numpy()


def metrics_and_rows(ds, pred_norm, norm):
    _, _, target_mean, target_std = norm
    raw = pred_norm * target_std[None, :] + target_mean[None, :]

    pred_a1 = raw[:, 0].astype(np.float64)
    pred_m = raw[:, 1].astype(np.float64)
    pred_gamma = np.exp(np.clip(raw[:, 2].astype(np.float64), -50.0, 50.0))

    true_a1 = ds.y_raw[:, 0].astype(np.float64)
    true_m = ds.y_raw[:, 1].astype(np.float64)
    true_gamma = np.exp(ds.y_raw[:, 2].astype(np.float64))

    err_a1 = np.abs(pred_a1 - true_a1)
    err_m = np.abs(pred_m - true_m)
    err_gamma = np.abs(pred_gamma - true_gamma)
    gamma_factor = np.maximum(
        pred_gamma / np.maximum(true_gamma, 1e-300),
        true_gamma / np.maximum(pred_gamma, 1e-300),
    )

    pass_a1 = err_a1 <= RECOVERY["a1_abs"]
    pass_m = err_m <= RECOVERY["m_abs"]
    pass_gamma = gamma_factor <= RECOVERY["gamma_factor"]
    pass_joint = pass_a1 & pass_m & pass_gamma

    def stats(e, prefix):
        return {
            f"{prefix}_mae": float(np.mean(e)),
            f"{prefix}_rmse": float(np.sqrt(np.mean(e ** 2))),
            f"{prefix}_median": float(np.median(e)),
            f"{prefix}_p90": float(np.quantile(e, 0.90)),
            f"{prefix}_p95": float(np.quantile(e, 0.95)),
        }

    out = {
        "n": len(ds),
        "R_a1": float(pass_a1.mean()),
        "R_m": float(pass_m.mean()),
        "R_gamma": float(pass_gamma.mean()),
        "R_joint": float(pass_joint.mean()),
        "gamma_factor_median": float(np.median(gamma_factor)),
        "gamma_factor_p90": float(np.quantile(gamma_factor, 0.90)),
        "gamma_factor_p95": float(np.quantile(gamma_factor, 0.95)),
    }
    out.update(stats(err_a1, "a1"))
    out.update(stats(err_m, "m"))
    out.update(stats(err_gamma, "gamma"))

    # Useful shrinkage diagnostics.
    for name, t, p in [
        ("a1", true_a1, pred_a1),
        ("m", true_m, pred_m),
        ("gamma", true_gamma, pred_gamma),
    ]:
        slope, intercept = np.polyfit(t, p, 1)
        out[f"{name}_pred_vs_true_slope"] = float(slope)
        out[f"{name}_pred_vs_true_intercept"] = float(intercept)
        out[f"{name}_corr"] = float(np.corrcoef(t, p)[0, 1])

    rows = {
        "true_a1": true_a1, "pred_a1": pred_a1,
        "true_m": true_m, "pred_m": pred_m,
        "true_gamma": true_gamma, "pred_gamma": pred_gamma,
        "a1_abs_error": err_a1, "m_abs_error": err_m,
        "gamma_abs_error": err_gamma, "gamma_factor_error": gamma_factor,
        "pass_a1": pass_a1, "pass_m": pass_m,
        "pass_gamma": pass_gamma, "joint_pass": pass_joint,
    }
    return out, rows


def save_predictions(path, ds, rows, seed):
    fields = [
        "physical_state_id",
        "true_a1", "pred_a1", "true_m", "pred_m", "true_gamma", "pred_gamma",
        "a1_abs_error", "m_abs_error", "gamma_abs_error", "gamma_factor_error",
        "pass_a1", "pass_m", "pass_gamma", "joint_pass",
        "continuous_margin", "noise_level", "seed",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(ds)):
            rec = {"physical_state_id": ds.ids[i]}
            for k, arr in rows.items():
                v = arr[i]
                if k.startswith("pass_") or k == "joint_pass":
                    v = int(v)
                rec[k] = v
            rec["continuous_margin"] = ds.margin[i]
            rec["noise_level"] = ds.noise_level[i]
            rec["seed"] = seed
            w.writerow(rec)


def verify_splits(train_ds, val_ds, tests):
    tr, va, te = set(train_ds.ids), set(val_ds.ids), set(tests["0pct"].ids)
    if tr & va or tr & te or va & te:
        raise RuntimeError("physical_state_id leakage across train/val/test")
    base = tests["0pct"].ids
    for name, ds in tests.items():
        if not np.array_equal(base, ds.ids):
            raise RuntimeError(f"test physical_state_id mismatch at {name}")
        if ds.x.shape[1] != train_ds.x.shape[1]:
            raise RuntimeError(f"input length mismatch at {name}")


def aggregate_results(results):
    metric_keys = [
        "R_a1", "R_m", "R_gamma", "R_joint",
        "a1_mae", "m_mae", "gamma_mae",
        "a1_p90", "m_p90", "gamma_p90", "gamma_factor_p90",
        "a1_pred_vs_true_slope", "m_pred_vs_true_slope", "gamma_pred_vs_true_slope",
    ]
    agg = {}
    for noise in ("0pct", "0p2pct", "1pct"):
        agg[noise] = {}
        for k in metric_keys:
            vals = np.asarray([r["tests"][noise][k] for r in results], np.float64)
            agg[noise][k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cp02_ml_hybrid_domain_v2")
    ap.add_argument("--output-dir", default="runs_cp02_tcn_v3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260920, 20260921, 20260922])
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--dropout", type=float, default=0.03)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--channels", type=int, nargs="+", default=[64, 96, 128, 160])
    ap.add_argument("--pool-bins", type=int, default=12)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every-batches", type=int, default=10)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--no-shortcut", action="store_true",
                    help="Pure no-norm TCN ablation: disable direct observation shortcut.")
    args = ap.parse_args()

    root = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    norm = load_norm(root / "normalization_train_only.npz")
    train_ds = CP02Dataset(root / "noise_0p2pct" / "train.npz", norm, root)
    val_ds = CP02Dataset(root / "noise_0p2pct" / "val.npz", norm, root)
    tests = {
        "0pct": CP02Dataset(root / "noise_0pct" / "test.npz", norm, root),
        "0p2pct": CP02Dataset(root / "noise_0p2pct" / "test.npz", norm, root),
        "1pct": CP02Dataset(root / "noise_1pct" / "test.npz", norm, root),
    }
    verify_splits(train_ds, val_ds, tests)

    q2 = train_ds.q2
    if q2 is None:
        for ds in [val_ds, *tests.values()]:
            if ds.q2 is not None:
                q2 = ds.q2
                break
    coord, coord_mode = make_coordinate_channels(q2, train_ds.x.shape[1])

    device = choose_device(args.device)
    backend = configure_backend(device, args.deterministic, not args.no_tf32)
    amp_enabled = device.type == "cuda" and not args.no_amp

    print(
        f"Loaded train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(tests['0pct'])}, input_length={train_ds.x.shape[1]}",
        flush=True,
    )
    print(f"coordinate_mode={coord_mode}", flush=True)
    print(
        f"device={device} amp={amp_enabled} tf32={backend['tf32']} "
        f"benchmark={backend['cudnn_benchmark']} shortcut={not args.no_shortcut}",
        flush=True,
    )

    # Dataset is small enough to keep all normalized tensors resident on GPU.
    train_x = torch.from_numpy(train_ds.x).to(device)
    train_y = torch.from_numpy(train_ds.y).to(device)
    val_x = torch.from_numpy(val_ds.x).to(device)
    val_y = torch.from_numpy(val_ds.y).to(device)
    test_x = {k: torch.from_numpy(v.x).to(device) for k, v in tests.items()}

    manifest = {
        "version": "cp02-tcn-amp-preserve-v3",
        "data_dir": str(root),
        "input_length": int(train_ds.x.shape[1]),
        "coordinate_mode": coord_mode,
        "architecture_changes": [
            "removed GroupNorm",
            "positional adaptive avg/max pooling",
            "direct observation shortcut",
            "global trace amplitude statistics",
        ],
        "shortcut_enabled": not args.no_shortcut,
        "pool_bins": args.pool_bins,
        "channels": args.channels,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "checkpoint_rule": "lexicographic max(min val recovery, joint val recovery, -val loss)",
        "recovery": RECOVERY,
        "seeds": args.seeds,
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    results = []
    for seed in args.seeds:
        seed_all(seed)
        model = AmpPreserveTCN3P(
            input_length=train_ds.x.shape[1],
            coord_channels=coord,
            channels=tuple(args.channels),
            kernel=args.kernel_size,
            dropout=args.dropout,
            pool_bins=args.pool_bins,
            use_shortcut=not args.no_shortcut,
        ).to(device)

        lossfn = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(4, args.patience // 4), min_lr=1e-6
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        # Recovery-aligned validation checkpoint.
        best_key = (-1.0, -1.0, -float("inf"))
        best_epoch = 0
        best_val_loss = float("inf")
        stale = 0
        history = []

        print(f"--- seed={seed} ---", flush=True)
        for ep in range(1, args.epochs + 1):
            tr_loss, sec, sps = train_epoch(
                model, train_x, train_y, lossfn, optimizer, scaler, amp_enabled,
                args.batch_size, seed, ep, args.log_every_batches
            )
            val_loss, val_pred = eval_loss_and_pred(
                model, val_x, val_y, lossfn, amp_enabled, args.batch_size
            )
            vr = recovery_from_pred_norm(val_pred, val_ds, norm)
            scheduler.step(val_loss)

            # Lexicographic rule: prioritize worst recovered parameter, then joint,
            # then continuous val loss.
            key = (vr["min_R"], vr["R_joint"], -val_loss)
            improved = key > best_key
            if improved:
                best_key = key
                best_epoch = ep
                best_val_loss = val_loss
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "seed": seed,
                        "best_epoch": ep,
                        "best_val_loss": val_loss,
                        "best_val_recovery": vr,
                        "input_length": train_ds.x.shape[1],
                        "channels": args.channels,
                        "pool_bins": args.pool_bins,
                        "shortcut_enabled": not args.no_shortcut,
                    },
                    seed_dir / "best.pt",
                )
            else:
                stale += 1

            lr_now = optimizer.param_groups[0]["lr"]
            history.append([
                ep, tr_loss, val_loss, lr_now, sec, sps,
                vr["R_a1"], vr["R_m"], vr["R_gamma"], vr["R_joint"], vr["min_R"],
                int(improved),
            ])

            print(
                f"ep={ep} DONE train={tr_loss:.6f} val={val_loss:.6f} "
                f"R=({100*vr['R_a1']:.1f},{100*vr['R_m']:.1f},{100*vr['R_gamma']:.1f})% "
                f"joint={100*vr['R_joint']:.1f}% "
                f"time={sec:.1f}s speed={sps:.0f}/s best_ep={best_epoch}",
                flush=True,
            )

            if stale >= args.patience:
                print(f"early stop at {ep}; best_epoch={best_epoch}", flush=True)
                break

        with open(seed_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch","train_loss","val_loss","lr","epoch_seconds","train_samples_per_sec",
                "val_R_a1","val_R_m","val_R_gamma","val_R_joint","val_min_R","checkpoint_saved"
            ])
            w.writerows(history)

        ckpt = safe_torch_load(seed_dir / "best.pt", device)
        model.load_state_dict(ckpt["model_state"])

        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_recovery": ckpt["best_val_recovery"],
            "tests": {},
        }

        for noise, ds in tests.items():
            pred = predict(model, test_x[noise], amp_enabled, args.batch_size)
            metrics, rows = metrics_and_rows(ds, pred, norm)
            result["tests"][noise] = metrics
            save_predictions(seed_dir / f"test_{noise}_predictions.csv", ds, rows, seed)
            print(
                f"{noise}: R_a1={100*metrics['R_a1']:.2f}% "
                f"R_m={100*metrics['R_m']:.2f}% "
                f"R_gamma={100*metrics['R_gamma']:.2f}% "
                f"joint={100*metrics['R_joint']:.2f}% "
                f"a1_slope={metrics['a1_pred_vs_true_slope']:.3f}",
                flush=True,
            )

        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        results.append(result)

        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    agg = aggregate_results(results)
    with open(out / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("=== aggregate ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
