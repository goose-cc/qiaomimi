from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import argparse, csv, json, math, random, time

import numpy as np
import torch
import torch.nn as nn

PARAM_NAMES = ["a1", "a2", "a3", "m", "gamma"]
TARGET_NAMES = ["a1", "a2", "a3", "m", "log_gamma"]
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
    im = np.asarray(z["input_mean"], np.float32)
    istd = np.maximum(np.asarray(z["input_std"], np.float32), 1e-12)
    tm = np.asarray(z["target_mean"], np.float32)
    ts = np.maximum(np.asarray(z["target_std"], np.float32), 1e-12)
    names = [str(x) for x in np.asarray(z["target_names"]).tolist()] if "target_names" in z.files else None
    transforms = [str(x) for x in np.asarray(z["target_transform"]).tolist()] if "target_transform" in z.files else None
    z.close()
    if tm.shape != (5,) or ts.shape != (5,):
        raise ValueError(f"target normalization must be shape (5,), got {tm.shape}/{ts.shape}")
    if names is not None and names != TARGET_NAMES:
        raise ValueError(f"Unexpected target_names={names}; expected {TARGET_NAMES}")
    if transforms is not None and transforms != ["identity", "identity", "identity", "identity", "natural_log"]:
        raise ValueError(f"Unexpected target_transform={transforms}")
    return im, istd, tm, ts


class FivePDataset:
    def __init__(self, path: Path, norm):
        self.path = Path(path)
        im, istd, tm, ts = norm
        z = np.load(self.path, allow_pickle=False)
        self.x_raw = np.asarray(z["gy"], np.float32)
        p = np.asarray(z["parameters"], np.float32)
        self.ids = np.asarray(z["physical_state_id"]).astype(str)
        z.close()
        if self.x_raw.ndim != 2:
            raise ValueError(f"gy must be [N,L], got {self.x_raw.shape}")
        if p.ndim != 2 or p.shape[1] != 5:
            raise ValueError(f"parameters must be [N,5], got {p.shape}")
        if len(p) != len(self.x_raw) or len(self.ids) != len(p):
            raise ValueError("gy/parameters/physical_state_id length mismatch")
        if np.any(p[:, 4] <= 0):
            raise ValueError("gamma must be > 0")
        self.parameters = p
        self.y_raw = p.copy()
        self.y_raw[:, 4] = np.log(self.y_raw[:, 4])
        self.x = ((self.x_raw - im) / istd).astype(np.float32)
        self.y = ((self.y_raw - tm) / ts).astype(np.float32)
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{path}: NaN/Inf found")

    def __len__(self):
        return len(self.x)


def verify_train_val(train_ds: FivePDataset, val_ds: FivePDataset):
    overlap = set(train_ds.ids) & set(val_ds.ids)
    if overlap:
        raise RuntimeError(f"physical_state_id leakage across train/val: {len(overlap)} states")
    if train_ds.x.shape[1] != val_ds.x.shape[1]:
        raise RuntimeError("train/val input length mismatch")


def make_coordinate_channels(q2: np.ndarray | None, length: int):
    if q2 is None:
        idx = np.linspace(-1.0, 1.0, length, dtype=np.float32)
        return torch.from_numpy(idx[None, :]), "index_fallback"
    q2 = np.asarray(q2, np.float64).reshape(-1)
    if len(q2) != length:
        raise ValueError(f"q2 length {len(q2)} != input length {length}")
    qmin, qmax = float(q2.min()), float(q2.max())
    qlin = 2.0 * (q2 - qmin) / max(qmax - qmin, 1e-30) - 1.0
    logabs = np.log(np.maximum(np.abs(q2), 1e-12))
    lmin, lmax = float(logabs.min()), float(logabs.max())
    logabs = 2.0 * (logabs - lmin) / max(lmax - lmin, 1e-30) - 1.0
    arr = np.stack([qlin, logabs], axis=0).astype(np.float32)
    return torch.from_numpy(arr), "q2_linear_plus_logabs"


class ResidualTCNBlockNoNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.03):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("Use an odd kernel size")
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


class FivePTCN(nn.Module):
    """Five-output version of the CP02 amplitude-preserving TCN v3."""
    def __init__(self, input_length: int, coord_channels: torch.Tensor,
                 channels=(64, 96, 128, 160), kernel=5, dropout=0.03,
                 pool_bins=12, shortcut_dim=128):
        super().__init__()
        self.input_length = int(input_length)
        self.register_buffer("coord", coord_channels[None, :, :].float(), persistent=True)
        blocks = []
        in_ch = 1 + coord_channels.shape[0]
        for i, out_ch in enumerate(channels):
            blocks.append(ResidualTCNBlockNoNorm(in_ch, out_ch, kernel, 2 ** i, dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(pool_bins)
        tcn_dim = 2 * channels[-1] * pool_bins
        self.obs_shortcut = nn.Sequential(
            nn.Linear(self.input_length, 192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, shortcut_dim), nn.GELU(),
        )
        stats_dim = 7
        self.fusion = nn.Sequential(
            nn.Linear(tcn_dim + shortcut_dim + stats_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 160), nn.GELU(),
        )
        def head():
            return nn.Sequential(nn.Linear(160, 80), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(80, 1))
        self.heads = nn.ModuleList([head() for _ in range(5)])

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
        c = self.coord.expand(x.shape[0], -1, -1)
        h = self.backbone(torch.cat([x.unsqueeze(1), c], dim=1))
        parts = [self.avg_pool(h).flatten(1), self.max_pool(h).flatten(1), self.trace_stats(x), self.obs_shortcut(x)]
        fused = self.fusion(torch.cat(parts, dim=1))
        return torch.cat([head(fused) for head in self.heads], dim=1)


class FivePMLP(nn.Module):
    def __init__(self, n_in: int, hidden=(512, 256, 128), dropout=0.05):
        super().__init__()
        layers = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)
        def head():
            mid = max(32, d // 2)
            return nn.Sequential(nn.Linear(d, mid), nn.GELU(), nn.Linear(mid, 1))
        self.heads = nn.ModuleList([head() for _ in range(5)])

    def forward(self, x):
        h = self.backbone(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def weighted_smooth_l1(pred, target, weights):
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights[None, :]).mean()


def train_epoch(model, x, y, optimizer, scaler, amp_enabled, batch_size, weights):
    model.train()
    perm = torch.randperm(x.shape[0], device=x.device)
    total = 0.0
    for s in range(0, x.shape[0], batch_size):
        idx = perm[s:s + batch_size]
        xb, yb = x.index_select(0, idx), y.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled):
            pred = model(xb)
            loss = weighted_smooth_l1(pred, yb, weights)
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
        total += float(loss.detach().float().item()) * len(idx)
    return total / x.shape[0]


@torch.no_grad()
def eval_loss_pred(model, x, y, amp_enabled, batch_size, weights):
    model.eval()
    total = 0.0
    preds = []
    for s in range(0, x.shape[0], batch_size):
        xb, yb = x[s:s + batch_size], y[s:s + batch_size]
        with amp_context(amp_enabled):
            p = model(xb)
            loss = weighted_smooth_l1(p, yb, weights)
        total += float(loss.detach().float().item()) * len(xb)
        preds.append(p.float().cpu())
    return total / x.shape[0], torch.cat(preds, dim=0).numpy()


def pred_norm_to_physical(pred_norm, norm):
    _, _, tm, ts = norm
    raw = pred_norm * ts[None, :] + tm[None, :]
    physical = raw.astype(np.float64)
    physical[:, 4] = np.exp(np.clip(physical[:, 4], -50.0, 50.0))
    return physical


def compute_metrics(ds: FivePDataset, pred_norm, norm):
    pred = pred_norm_to_physical(pred_norm, norm)
    true = ds.parameters.astype(np.float64)
    abs_err = np.abs(pred - true)
    gf = np.maximum(pred[:, 4] / np.maximum(true[:, 4], 1e-300), true[:, 4] / np.maximum(pred[:, 4], 1e-300))
    pass_a1 = abs_err[:, 0] <= RECOVERY["a1_abs"]
    pass_m = abs_err[:, 3] <= RECOVERY["m_abs"]
    pass_gamma = gf <= RECOVERY["gamma_factor"]
    joint = pass_a1 & pass_m & pass_gamma
    out = {
        "n": len(ds),
        "R_a1": float(pass_a1.mean()),
        "R_m": float(pass_m.mean()),
        "R_gamma": float(pass_gamma.mean()),
        "R_joint": float(joint.mean()),
        "min_R": float(min(pass_a1.mean(), pass_m.mean(), pass_gamma.mean())),
        "gamma_factor_median": float(np.median(gf)),
        "gamma_factor_p90": float(np.quantile(gf, 0.90)),
        "gamma_factor_p95": float(np.quantile(gf, 0.95)),
    }
    for j, name in enumerate(PARAM_NAMES):
        e = abs_err[:, j]
        out[f"{name}_mae"] = float(e.mean())
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{name}_p90"] = float(np.quantile(e, 0.90))
        out[f"{name}_p95"] = float(np.quantile(e, 0.95))
        if np.std(true[:, j]) > 0 and np.std(pred[:, j]) > 0:
            out[f"{name}_corr"] = float(np.corrcoef(true[:, j], pred[:, j])[0, 1])
        else:
            out[f"{name}_corr"] = float("nan")
    rows = {"true": true, "pred": pred, "abs_err": abs_err, "gamma_factor": gf,
            "pass_a1": pass_a1, "pass_m": pass_m, "pass_gamma": pass_gamma, "joint": joint}
    return out, rows


def save_predictions(path: Path, ds: FivePDataset, rows, seed: int, model_name: str):
    fields = ["physical_state_id"]
    for n in PARAM_NAMES:
        fields += [f"true_{n}", f"pred_{n}", f"{n}_abs_error"]
    fields += ["gamma_factor_error", "pass_a1", "pass_m", "pass_gamma", "joint_pass", "seed", "model"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(ds)):
            rec = {"physical_state_id": ds.ids[i], "gamma_factor_error": rows["gamma_factor"][i],
                   "pass_a1": int(rows["pass_a1"][i]), "pass_m": int(rows["pass_m"][i]),
                   "pass_gamma": int(rows["pass_gamma"][i]), "joint_pass": int(rows["joint"][i]),
                   "seed": seed, "model": model_name}
            for j, n in enumerate(PARAM_NAMES):
                rec[f"true_{n}"] = rows["true"][i, j]
                rec[f"pred_{n}"] = rows["pred"][i, j]
                rec[f"{n}_abs_error"] = rows["abs_err"][i, j]
            w.writerow(rec)


def build_model(name: str, input_length: int, coord, args):
    if name == "tcn":
        return FivePTCN(input_length, coord, tuple(args.tcn_channels), args.kernel_size,
                        args.tcn_dropout, args.pool_bins)
    if name == "mlp":
        return FivePMLP(input_length, tuple(args.mlp_hidden), args.mlp_dropout)
    raise ValueError(name)


def train_one(model_name, seed, train_ds, val_ds, norm, coord, device, amp_enabled, args, out_root):
    seed_all(seed)
    model = build_model(model_name, train_ds.x.shape[1], coord, args).to(device)
    lr = args.tcn_lr if model_name == "tcn" else args.mlp_lr
    wd = args.tcn_weight_decay if model_name == "tcn" else args.mlp_weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(4, args.patience // 4), min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_x = torch.from_numpy(train_ds.x).to(device)
    train_y = torch.from_numpy(train_ds.y).to(device)
    val_x = torch.from_numpy(val_ds.x).to(device)
    val_y = torch.from_numpy(val_ds.y).to(device)
    weights = torch.tensor(args.target_weights, dtype=torch.float32, device=device)

    run_dir = out_root / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_key = (-1.0, -1.0, -float("inf"))
    best_epoch = 0
    stale = 0
    history = []
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_x, train_y, optimizer, scaler, amp_enabled, args.batch_size, weights)
        val_loss, val_pred = eval_loss_pred(model, val_x, val_y, amp_enabled, args.batch_size, weights)
        metrics, _ = compute_metrics(val_ds, val_pred, norm)
        scheduler.step(val_loss)
        key = (metrics["min_R"], metrics["R_joint"], -val_loss)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = ep
            stale = 0
            torch.save({
                "model_state": model.state_dict(), "model": model_name, "seed": seed,
                "best_epoch": ep, "best_val_loss": val_loss, "best_val_metrics": metrics,
                "input_length": train_ds.x.shape[1], "target_names": TARGET_NAMES,
                "recovery": RECOVERY,
            }, run_dir / "best.pt")
        else:
            stale += 1
        history.append([ep, tr_loss, val_loss, optimizer.param_groups[0]["lr"],
                        metrics["R_a1"], metrics["R_m"], metrics["R_gamma"], metrics["R_joint"], metrics["min_R"], int(improved)])
        if ep == 1 or ep % args.log_every == 0 or improved:
            print(f"[{model_name} seed={seed}] ep={ep:03d} train={tr_loss:.6f} val={val_loss:.6f} "
                  f"R=({100*metrics['R_a1']:.1f},{100*metrics['R_m']:.1f},{100*metrics['R_gamma']:.1f})% "
                  f"joint={100*metrics['R_joint']:.1f}% best={best_epoch}", flush=True)
        if stale >= args.patience:
            break

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","lr","val_R_a1","val_R_m","val_R_gamma","val_R_joint","val_min_R","checkpoint_saved"])
        w.writerows(history)

    ckpt = safe_torch_load(run_dir / "best.pt", device)
    model.load_state_dict(ckpt["model_state"])
    val_loss, val_pred = eval_loss_pred(model, val_x, val_y, amp_enabled, args.batch_size, weights)
    metrics, rows = compute_metrics(val_ds, val_pred, norm)
    save_predictions(run_dir / "val_predictions.csv", val_ds, rows, seed, model_name)
    result = {"model": model_name, "seed": seed, "best_epoch": best_epoch,
              "best_val_loss": float(val_loss), "val": metrics,
              "elapsed_seconds": float(time.perf_counter() - t0)}
    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    del model, optimizer, scheduler, scaler, train_x, train_y, val_x, val_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate(results, out_root: Path):
    rows = []
    summary = {}
    for model in sorted(set(r["model"] for r in results)):
        subset = [r for r in results if r["model"] == model]
        summary[model] = {}
        keys = ["R_a1","R_m","R_gamma","R_joint","a1_mae","a2_mae","a3_mae","m_mae","gamma_mae","gamma_factor_p90"]
        for k in keys:
            vals = np.asarray([r["val"][k] for r in subset], np.float64)
            summary[model][k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals)>1 else 0.0)}
        rows.append({"model": model, **{k: summary[model][k]["mean"] for k in keys}})
    with open(out_root / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_root / "comparison_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Five-parameter Train/Val comparison: TCN vs MLP. Never loads Test.")
    ap.add_argument("--data-dir", required=True, help="Directory containing noise_0p2pct/ and normalization_train_only.npz")
    ap.add_argument("--output-dir", default="runs_fivep_compare_v1")
    ap.add_argument("--models", nargs="+", choices=["tcn","mlp"], default=["tcn","mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260920,20260921,20260922])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--target-weights", type=float, nargs=5, default=[1,1,1,1,1], metavar=("A1","A2","A3","M","GAMMA"))
    ap.add_argument("--tcn-lr", type=float, default=5e-4)
    ap.add_argument("--tcn-weight-decay", type=float, default=3e-5)
    ap.add_argument("--tcn-dropout", type=float, default=0.03)
    ap.add_argument("--tcn-channels", type=int, nargs="+", default=[64,96,128,160])
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--pool-bins", type=int, default=12)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    ap.add_argument("--mlp-dropout", type=float, default=0.05)
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[512,256,128])
    args = ap.parse_args()

    root = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in [root / "noise_0p2pct" / "train.npz", root / "noise_0p2pct" / "val.npz", root / "normalization_train_only.npz"]:
        if not p.exists():
            raise FileNotFoundError(p)
    # Explicit guard: training script must not need or inspect Test.
    norm = load_norm(root / "normalization_train_only.npz")
    train_ds = FivePDataset(root / "noise_0p2pct" / "train.npz", norm)
    val_ds = FivePDataset(root / "noise_0p2pct" / "val.npz", norm)
    verify_train_val(train_ds, val_ds)
    q2_path = root / "q2.npy"
    q2 = np.load(q2_path, allow_pickle=False) if q2_path.exists() else None
    coord, coord_mode = make_coordinate_channels(q2, train_ds.x.shape[1])
    device = choose_device(args.device)
    backend = configure_backend(device, args.deterministic, not args.no_tf32)
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"Loaded Train={len(train_ds)} Val={len(val_ds)} input={train_ds.x.shape[1]} targets=5")
    print(f"Models={args.models} device={device} amp={amp_enabled} coord={coord_mode}")

    manifest = {
        "experiment": "fivep-blind-v1-train-val-model-compare",
        "test_access": "NONE - this script loads Train/Val only",
        "models": args.models, "seeds": args.seeds, "epochs": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience,
        "target_names": TARGET_NAMES,
        "target_transform": ["identity","identity","identity","identity","natural_log"],
        "target_weights": args.target_weights,
        "recovery": RECOVERY,
        "checkpoint_rule": "lexicographic max(min(R_a1,R_m,R_gamma), R_joint, -val_loss)",
        "tcn": {"channels": args.tcn_channels, "kernel": args.kernel_size, "pool_bins": args.pool_bins,
                "dropout": args.tcn_dropout, "lr": args.tcn_lr, "weight_decay": args.tcn_weight_decay,
                "architecture": "CP02 amplitude-preserving TCN v3 adapted to 5 independent heads"},
        "mlp": {"hidden": args.mlp_hidden, "dropout": args.mlp_dropout, "lr": args.mlp_lr,
                "weight_decay": args.mlp_weight_decay, "architecture": "shared MLP backbone + 5 independent heads"},
        "backend": backend,
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    results = []
    for model_name in args.models:
        for seed in args.seeds:
            results.append(train_one(model_name, seed, train_ds, val_ds, norm, coord, device, amp_enabled, args, out))
    summary = aggregate(results, out)
    print("\n=== Train/Val comparison (NO TEST) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
