"""Create separate five-parameter trainer and private evaluator ZIPs.

Run from the repository root. Read-only on all original project files.
The trainer ZIP contains only Train/Val examples and dependencies;
all test labels remain in the private evaluator ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

SOURCES = (
    "build_fivep_dataset.py", "validate_fivep_dataset.py",
    "build_fivep_codebook.py", "build_fivep_blind.py",
    "validate_fivep_blind.py", "test_fivep_blind.py",
    "mc_physics.py", "mc_pool_config.py", "requirements.txt",
)
NAMES = ("a1", "a2", "a3", "m", "gamma")
BOUNDS = ((0.05, 0.20), (0.0, 0.05), (-0.05, 0.05),
          (0.40, 1.20), (0.01, 1.00))
GATES = {"a1_abs": 0.005, "m_abs": 0.03, "gamma_factor": 1.2}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        x = z["gy"].copy()
        y = z["parameters"].copy()
        state_ids = z["physical_state_id"].copy()
    if x.ndim != 2 or x.shape[1] != 100 or y.shape != (len(x), 5):
        raise ValueError(f"Invalid data dimensions: {path}")
    if state_ids.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Bad IDs or non-finite data: {path}")
    lo, hi = np.array(BOUNDS, dtype=float).T
    if np.any(y < lo - 1e-6) or np.any(y > hi + 1e-6):
        raise ValueError(f"Out-of-domain labels: {path}")
    return x, y, state_ids


def minimal_npz(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, gy=x, parameters=y, physical_state_id=ids)
    return buf.getvalue()


def make_zip(path: Path, contents: dict[str, bytes]) -> None:
    hashes = {name: digest(blob) for name, blob in contents.items()}
    contents = dict(contents)
    contents["SHA256_MANIFEST.json"] = json.dumps(
        hashes, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for name, blob in contents.items():
            z.writestr(name, blob)
    with ZipFile(path) as z:
        assert z.testzip() is None
        assert all(digest(z.read(k)) == v for k, v in hashes.items())


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", default="data_fivep_blind_v1")
    cli.add_argument("--out", default="fivep_handoff_v1")
    args = cli.parse_args()

    root = Path(__file__).resolve().parent
    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"STOP: output already exists; will not overwrite: {out}")
    required_data = (
        "metadata.json", "q2.npy", "normalization_train_only.npz",
        "noise_0p2pct/train.npz", "noise_0p2pct/val.npz",
        "noise_0p2pct/test.npz", "blind_validation_report.json",
        "validation_report.json", "state_manifest.csv",
    )
    missing = [p for p in SOURCES if not (root / p).is_file()]
    missing += [p for p in required_data if not (data / p).is_file()]
    if missing:
        raise SystemExit("STOP: missing required source/data: " + ", ".join(missing))

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if (meta.get("dataset_kind") != "FIVEP_INDEPENDENT_HELDOUT_V1"
            or tuple(meta.get("parameter_names", [])) != NAMES
            or tuple(meta.get("q2_domain", [])) != (-100.0, -6.0)
            or meta.get("q2_points") != 100
            or meta.get("reference_noise") != 0.002
            or meta.get("recovery_tolerances") != GATES
            or meta.get("blind_protocol", {}).get("validation_test_g_not_used_to_select_states") is not True):
        raise ValueError("STOP: unexpected physics/recovery/holdout contract")
    if any(tuple(meta["parameter_ranges"].get(n, [])) != b
           for n, b in zip(NAMES, BOUNDS)):
        raise ValueError("STOP: parameter bounds differ from frozen contract")

    for src, recorded in meta.get("source_sha256", {}).items():
        if src in SOURCES and digest((root / src).read_bytes()) != recorded:
            raise ValueError(f"STOP: source does not match validated metadata: {src}")

    q2 = np.load(data / "q2.npy", allow_pickle=False)
    if q2.shape != (100,) or not np.isfinite(q2).all() or q2[0] != -100 or q2[-1] != -6:
        raise ValueError("STOP: invalid q2 grid")
    if digest(q2.tobytes()) != meta.get("q2_sha256"):
        raise ValueError("STOP: q2 grid checksum mismatch")
    with np.load(data / "normalization_train_only.npz", allow_pickle=False) as norm:
        if norm["input_mean"].shape != (100,) or norm["input_std"].shape != (100,):
            raise ValueError("STOP: unexpected input normalization dimensions")
        if norm["target_mean"].shape != (5,) or norm["target_std"].shape != (5,):
            raise ValueError("STOP: unexpected target normalization dimensions")
        if (tuple(norm["target_names"].astype(str))[-1] != "log_gamma"
                or tuple(norm["target_transform"].astype(str))[-1] != "natural_log"):
            raise ValueError("STOP: gamma target transform changed")

    items = {}
    for split in ("train", "val", "test"):
        items[split] = safe_npz(data / "noise_0p2pct" / f"{split}.npz")
    for a, b in combinations(items, 2):
        ids_a, ids_b = set(items[a][2].tolist()), set(items[b][2].tolist())
        if not ids_a.isdisjoint(ids_b):
            raise ValueError(f"STOP: state ID leakage between {a} and {b}")
        truths_a = set(map(tuple, np.round(items[a][1].astype(float), 8)))
        truths_b = set(map(tuple, np.round(items[b][1].astype(float), 8)))
        if not truths_a.isdisjoint(truths_b):
            raise ValueError(f"STOP: parameter truth leakage between {a} and {b}")

    train_readme = (
        "FIVE-PARAMETER UNKNOWN-STATE REGRESSION — TRAINER PACKAGE\n"
        "Input: data/noise_0p2pct/{train,val}.npz key 'gy', shape [N,100].\n"
        "Targets: key 'parameters', shape [N,5], order [a1,a2,a3,m,gamma].\n"
        "Use physical_state_id only to audit grouping; never as model input.\n"
        "Normalize only with provided normalization_train_only.npz; target\n"
        "5th coordinate uses log_gamma; invert before evaluating physical gamma.\n"
        "Only use Train to fit weights, Val for tuning. No Test truth included.\n"
        "Recovery gates: |da1|<=0.005; |dm|<=0.03; F_gamma<=1.2.\n"
        "Test set is independently chosen and held by the evaluator.\n"
        "DO NOT combine closed-set examples with this experiment for an unseen-state claim.\n"
        "Passing data checks does NOT certify trained model recovery.\n"
    ).encode("utf-8")
    trainer: dict[str, bytes] = {f"src/{s}": (root / s).read_bytes() for s in SOURCES}
    for split in ("train", "val"):
        trainer[f"data/noise_0p2pct/{split}.npz"] = minimal_npz(*items[split])
    for src in ("q2.npy", "normalization_train_only.npz"):
        trainer[f"data/{src}"] = (data / src).read_bytes()
    trainer["TRAINER_README.txt"] = train_readme
    trainer["CONTRACT.json"] = json.dumps(
        {"names": NAMES, "bounds": BOUNDS, "q2": [-100, -6],
         "q2_points": 100, "noise_rms": 0.002, "recovery_gates": GATES,
         "task": "unseen-state regression; test truth private"},
        indent=2, ensure_ascii=False
    ).encode("utf-8")

    evaluator: dict[str, bytes] = {}
    for item in sorted(data.rglob("*")):
        if item.is_file():
            evaluator[f"data/{item.relative_to(data).as_posix()}"] = item.read_bytes()
    evaluator["EVALUATOR_README.txt"] = (
        "PRIVATE: contains held-out test labels and reproducibility metadata.\n"
        "Do NOT give this archive to the model trainer before final freeze.\n"
        "Evaluate final predictions against data/noise_0p2pct/test.npz,\n"
        "and report each recovery gate plus their joint pass fraction.\n"
    ).encode("utf-8")

    out.mkdir(parents=True, exist_ok=False)
    training_path = out / "fivep_train_val_v1.zip"
    evaluation_path = out / "fivep_private_evaluator_v1.zip"
    make_zip(training_path, trainer)
    make_zip(evaluation_path, evaluator)
    (out / "ZIP_SHA256.txt").write_text(
        f"{digest(training_path.read_bytes())}  {training_path.name}\n"
        f"{digest(evaluation_path.read_bytes())}  {evaluation_path.name}\n",
        encoding="utf-8",
    )
    print("HANDOFF PACKAGES PASS")
    print(f"Trainer: {training_path}")
    print(f"Private evaluator (do NOT send to trainer): {evaluation_path}")
    print(f"Rows Train={len(items['train'][0])}, Val={len(items['val'][0])}, "
          f"Test(private)={len(items['test'][0])}")
    print("Original dataset and source files were not modified")


if __name__ == "__main__":
    main()
