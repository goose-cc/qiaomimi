"""Build a CLOSED-SET five-parameter noisy codebook without neural training.

The same selected physical states intentionally occur in train/val/test, but
independent noise is generated for each split. This tests recognition of a
finite codebook under unseen noise, NOT generalization to unseen parameter
states and NOT continuous-domain identifiability/recovery.

Requires the project's real mc_physics.py and the previously validated
five-parameter codebook dataset. Never overwrites the source or output.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mc_physics import (
    scaled_forward_observation_at_q2_numpy,
    valid_parameter_mask_numpy,
)

NQ = 100
Q2_LO, Q2_HI = -100.0, -6.0
PARAM_NAMES = ("a1", "a2", "a3", "m", "gamma")
LO = np.array([0.05, 0.0, -0.05, 0.40, 0.01], dtype=np.float64)
HI = np.array([0.20, 0.05, 0.05, 1.20, 1.00], dtype=np.float64)
NOISE = 0.002
MIN_D = 6.0
A1_GATE, M_GATE, GAMMA_GATE = 0.005, 0.03, 1.2


def get_selected_states(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Union the verified old train/val/test physical states, not noisy rows."""
    states = {}
    for split in ("train", "val", "test"):
        file = root / "noise_0p2pct" / f"{split}.npz"
        if not file.is_file():
            raise FileNotFoundError(f"Expected source file: {file}")
        with np.load(file, allow_pickle=False) as d:
            if not {"parameters", "physical_state_id", "gy"}.issubset(d.files):
                raise ValueError(f"Missing expected keys in {file}: {d.files}")
            p, sid, gy = d["parameters"], d["physical_state_id"], d["gy"]
            if p.ndim != 2 or p.shape[1] != 5 or gy.shape != (len(p), NQ):
                raise ValueError(f"Invalid five-parameter source shapes in {file}")
            if sid.shape != (len(p),):
                raise ValueError(f"Invalid physical_state_id in {file}")
            if not (np.isfinite(p).all() and np.isfinite(gy).all()):
                raise ValueError(f"Nonfinite source content in {file}")
            for key, pp in zip(sid.astype(str), p):
                cur = np.array(pp, dtype=np.float64)
                if key in states and not np.allclose(cur, states[key], rtol=0, atol=1e-7):
                    raise ValueError(f"Inconsistent parameter truth for state {key}")
                states[key] = cur
    if len(states) < 2:
        raise ValueError("Need at least two distinct physical states")
    keys = np.array(sorted(states), dtype=np.str_)
    parameters = np.stack([states[key] for key in keys], axis=0)
    return keys, parameters


def get_q2(root: Path) -> np.ndarray:
    """Use q2 from source where present; otherwise use known 100-point uniform grid."""
    candidates = []
    for file in (root / "q2.npy", root / "q2_grid.npy"):
        if file.exists():
            candidates.append(np.load(file, allow_pickle=False))
    for split in ("train", "val", "test"):
        file = root / "noise_0p2pct" / f"{split}.npz"
        with np.load(file, allow_pickle=False) as d:
            if "q2" in d.files:
                candidates.append(d["q2"])
    q2 = np.asarray(candidates[0] if candidates else np.linspace(Q2_LO, Q2_HI, NQ), dtype=np.float64).reshape(-1)
    if any(not np.array_equal(q2, np.asarray(c).reshape(-1)) for c in candidates[1:]):
        raise ValueError("Conflicting q2 grids in the source dataset")
    if q2.shape != (NQ,) or not np.isfinite(q2).all() or not np.all(np.diff(q2) > 0):
        raise ValueError("Invalid 100-point q2 grid")
    if not np.isclose(q2[0], Q2_LO, rtol=0, atol=1e-8) or not np.isclose(q2[-1], Q2_HI, rtol=0, atol=1e-8):
        raise ValueError("The original q2=[-100,-6] contract changed")
    return q2


def enforce_contract(p: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if p.shape[1:] != (5,) or not np.isfinite(p).all():
        raise ValueError("Invalid parameters")
    p = np.array(p, dtype=np.float64, copy=True)
    # Inputs may be float32. Snap only machine-roundoff representations
    # of exact endpoints (e.g. float32(0.2)) to the frozen bounds.
    eps = 1e-7
    if np.any(p < LO - eps) or np.any(p > HI + eps):
        raise ValueError("Parameter outside frozen five-dimensional bounds")
    for j in range(5):
        p[np.abs(p[:, j] - LO[j]) < eps, j] = LO[j]
        p[np.abs(p[:, j] - HI[j]) < eps, j] = HI[j]
        if not np.any(p[:, j] == LO[j]) or not np.any(p[:, j] == HI[j]):
            raise ValueError(f"Codebook lacks exact {PARAM_NAMES[j]} domain endpoint coverage")
    if not valid_parameter_mask_numpy(p).all():
        raise ValueError("At least one selected state fails continuous rho>=0")
    if q2.shape != (NQ,) or not np.array_equal(q2[[0, -1]], [Q2_LO, Q2_HI]):
        raise ValueError("q2 hard contract changed")
    return p


def min_distance(g: np.ndarray) -> float:
    norms = np.sum(g * g, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2 * (g @ g.T), 0.0)
    np.fill_diagonal(squared, np.inf)
    sigma = NOISE * np.sqrt(np.mean(g * g, axis=1))
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("Invalid g RMS or sigma")
    d = np.sqrt(squared) / np.maximum(sigma[:, None], sigma[None, :])
    return float(np.min(d))


def make_split(g: np.ndarray, p: np.ndarray, ids: np.ndarray, reps: int,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
    if reps < 1:
        raise ValueError("Noise replicas per state must be positive")
    count = len(p)
    labels = np.repeat(np.arange(count, dtype=np.int32), reps)
    noise_sigma = (NOISE * np.sqrt(np.mean(g * g, axis=1)))[labels]
    obs = g[labels] + noise_sigma[:, None] * rng.standard_normal((len(labels), NQ))
    perm = rng.permutation(len(labels))
    labels = labels[perm]
    return {
        "gy": obs[perm].astype(np.float32),
        "parameters": p[labels].astype(np.float32),
        "physical_state_id": ids[labels],
        "code_id": labels,
    }


def decode_using_train_only(train: dict, evaluate: dict) -> dict[str, float]:
    """No NN and no clean oracle: estimate prototypes using noisy train rows."""
    count = len(np.unique(train["code_id"]))
    mu = np.zeros((count, NQ), dtype=np.float64)
    for k in range(count):
        x = np.asarray(train["gy"][train["code_id"] == k], dtype=np.float64)
        if len(x) < 2:
            raise ValueError("Use >=2 training replicas per code for template estimation")
        mu[k] = x.mean(axis=0)
    sigma = NOISE * np.sqrt(np.mean(mu * mu, axis=1))
    if np.any(sigma <= 0):
        raise ValueError("Invalid template noise scales")
    x = np.asarray(evaluate["gy"], dtype=np.float64)
    predictions = []
    # Gaussian negative log-likelihood with state-dependent noise scale.
    # Template uncertainty from training replicas is a minor constant factor
    # when equal replica counts are used for all states.
    for start in range(0, len(x), 128):
        chunk = x[start:start + 128]
        z2 = np.sum(chunk * chunk, axis=1, keepdims=True)
        mu2 = np.sum(mu * mu, axis=1)
        distance2 = np.maximum(z2 + mu2 - 2 * (chunk @ mu.T), 0.0)
        scores = distance2 / (sigma[None, :] ** 2) + 2 * NQ * np.log(sigma[None, :])
        predictions.append(np.argmin(scores, axis=1))
    predicted = np.concatenate(predictions)
    true = evaluate["code_id"]
    inferred_p = np.asarray(train["parameters"])[np.array([np.flatnonzero(train["code_id"] == k)[0] for k in predicted])]
    truth_p = evaluate["parameters"]
    a1_ok = np.abs(inferred_p[:, 0] - truth_p[:, 0]) <= A1_GATE + 1e-9
    m_ok = np.abs(inferred_p[:, 3] - truth_p[:, 3]) <= M_GATE + 1e-9
    factor = np.maximum(inferred_p[:, 4] / truth_p[:, 4], truth_p[:, 4] / inferred_p[:, 4])
    gamma_ok = factor <= GAMMA_GATE + 1e-7
    return {
        "top1_code_accuracy": float(np.mean(predicted == true)),
        "a1_gate_fraction": float(np.mean(a1_ok)),
        "m_gate_fraction": float(np.mean(m_ok)),
        "gamma_gate_fraction": float(np.mean(gamma_ok)),
        "joint_gate_fraction": float(np.mean(a1_ok & m_ok & gamma_ok)),
    }


def build(data: Path, out: Path, train_reps: int, val_reps: int, test_reps: int,
          seed: int) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if train_reps < 2 or val_reps < 1 or test_reps < 1:
        raise ValueError("Require >=2 train and >=1 val/test noisy replicas per code")
    ids, p0 = get_selected_states(data)
    q2 = get_q2(data)
    p = enforce_contract(p0, q2)
    clean = scaled_forward_observation_at_q2_numpy(p, q2, integration_points=128,
                                                    output_dtype=np.float64)
    if clean.shape != (len(p), NQ) or not np.isfinite(clean).all():
        raise ValueError("Invalid physical forward g")
    dmin = min_distance(clean)
    if dmin < MIN_D:
        raise ValueError(f"Source codebook is insufficiently separated: Dmin={dmin:.6g} < {MIN_D}")
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    rng_train, rng_val, rng_test = [np.random.default_rng(ss) for ss in streams]
    train = make_split(clean, p, ids, train_reps, rng_train)
    val = make_split(clean, p, ids, val_reps, rng_val)
    test = make_split(clean, p, ids, test_reps, rng_test)
    metrics = {"val": decode_using_train_only(train, val),
               "test": decode_using_train_only(train, test)}
    report = {
        "contract": {
            "order": PARAM_NAMES, "lower": LO.tolist(), "upper": HI.tolist(),
            "q2_min": Q2_LO, "q2_max": Q2_HI, "n_q2": NQ, "relative_noise_rms": NOISE,
            "recovery_a1_abs": A1_GATE, "recovery_m_abs": M_GATE,
            "recovery_gamma_factor": GAMMA_GATE,
        },
        "task": "CLOSED_SET_FINITE_5P_RECOGNITION_NOISE_GENERALIZATION_ONLY",
        "source": str(data.resolve()), "seed": int(seed),
        "state_count": len(p), "min_codebook_distance": dmin,
        "splits": {"train": {"replicas_per_state": train_reps, "rows": len(train["gy"])},
                   "val": {"replicas_per_state": val_reps, "rows": len(val["gy"])},
                   "test": {"replicas_per_state": test_reps, "rows": len(test["gy"])}},
        "template_no_training_metrics": metrics,
        "caveat": "Same states deliberately in all splits. No unseen-state or continuous-domain recovery claim. Network was not trained.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="_5pclosed_", dir=out.parent))
    try:
        noisy = tmp / "noise_0p2pct"
        noisy.mkdir()
        for name, payload in (("train", train), ("val", val), ("test", test)):
            np.savez_compressed(noisy / f"{name}.npz", **payload)
        np.save(tmp / "q2.npy", q2)
        np.savez_compressed(tmp / "codebook_templates_physical.npz",
                            clean_gy=clean.astype(np.float32),
                            parameters=p.astype(np.float32),
                            physical_state_id=ids,
                            code_id=np.arange(len(p), dtype=np.int32))
        avg = train["gy"].astype(np.float64)
        np.savez_compressed(tmp / "train_normalization.npz",
                            mean=avg.mean(axis=0).astype(np.float32),
                            std=np.maximum(avg.std(axis=0), 1e-9).astype(np.float32))
        (tmp / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        # Verify independent draws and content before committing new directory.
        if np.array_equal(train["gy"][:min(3, len(train["gy"]))], test["gy"][:min(3, len(test["gy"]))]):
            raise RuntimeError("Train/test noise may not be independent")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Previously validated fivep codebook dataset")
    parser.add_argument("--out", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--train-reps", type=int, default=8)
    parser.add_argument("--val-reps", type=int, default=2)
    parser.add_argument("--test-reps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    r = build(args.data, args.out, args.train_reps, args.val_reps, args.test_reps, args.seed)
    print(f"CREATED {args.out} | distinct code states {r['state_count']} | Dmin {r['min_codebook_distance']:.5g}")
    print(f"Rows train/val/test: {[r['splits'][x]['rows'] for x in ('train', 'val', 'test')]}")
    for x in ("val", "test"):
        result = r["template_no_training_metrics"][x]
        print(f"{x.upper()} NO-NETWORK TRAIN-ONLY TEMPLATE: "
              f"top1={result['top1_code_accuracy']:.3%}, "
              f"joint_recovery={result['joint_gate_fraction']:.3%}, "
              f"a1={result['a1_gate_fraction']:.3%}, "
              f"m={result['m_gate_fraction']:.3%}, "
              f"gamma={result['gamma_gate_fraction']:.3%}")
    print("NOTE: finite closed-set / unseen noise only. NOT unseen-state or continuous-domain recovery.")
    print("No neural network trained; original dataset untouched.")


if __name__ == "__main__":
    main()
