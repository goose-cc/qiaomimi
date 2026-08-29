from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_core import (
    PARAMETER_NAMES,
    add_noise_and_interpolate,
    default_output_dir,
    load_config,
    noise_dir_name,
    parameter_matrix_from_rows,
    read_csv_rows,
    resolve_per_state_count,
    stable_state_seed,
    update_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand split physical states into noisy datasets")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--compressed", action="store_true")
    return p.parse_args()


def _save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def run(config_path: str, output_dir: str | None = None, compressed: bool | None = None) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    cand_rows = read_csv_rows(out / "candidate_states.csv")
    split_rows = read_csv_rows(out / "split_manifest.csv")
    cand_index = {r["state_id"]: int(r["state_index"]) for r in cand_rows}
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g_all = np.asarray(bank["g_clean"], dtype=np.float32)
    q2 = np.asarray(bank["q2"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int32)
    noise_cfg = cfg["noise"]
    levels = [float(x) for x in noise_cfg["noise_levels"]]
    base_seed = int(noise_cfg.get("seed", cfg["sampling"]["seed"] + 2000))
    compressed = bool(noise_cfg.get("compressed", True) if compressed is None else compressed)

    split_state_rows = {name: [r for r in split_rows if r["split"] == name] for name in ("train", "val", "test")}
    per_state = {
        name: resolve_per_state_count(noise_cfg, name, len(split_state_rows[name]))
        for name in ("train", "val", "test")
    }
    actual_counts: dict[str, dict[str, int]] = {}

    for level in levels:
        ndir = out / noise_dir_name(level)
        actual_counts[str(level)] = {}
        for split in ("train", "val", "test"):
            rows = split_state_rows[split]
            count = per_state[split]
            blocks: dict[str, list[np.ndarray]] = {
                "gy": [], "gy_noisy": [], "gy_clean": [], "parameters": [],
                "a1": [], "a2": [], "a3": [], "m": [], "gamma": [],
                "state_id": [], "realization_id": [], "noise_level": [], "noise_sigma": [],
            }
            for local_i, row in enumerate(rows):
                cidx = cand_index[row["state_id"]]
                p = np.asarray([float(row[name]) for name in PARAMETER_NAMES], dtype=np.float32)
                seed = stable_state_seed(base_seed, split, cidx, level)
                noisy_master, _, sigma = add_noise_and_interpolate(
                    g_all[cidx], q2, obs_idx, level, count, seed
                )
                blocks["gy"].append(noisy_master)
                blocks["gy_noisy"].append(noisy_master)
                blocks["gy_clean"].append(np.repeat(g_all[cidx][None, :], count, axis=0))
                blocks["parameters"].append(np.repeat(p[None, :], count, axis=0))
                for j, name in enumerate(PARAMETER_NAMES):
                    blocks[name].append(np.full(count, p[j], dtype=np.float32))
                sid_dtype = f"U{max(16, len(row['state_id']))}"
                blocks["state_id"].append(np.full(count, row["state_id"], dtype=sid_dtype))
                blocks["realization_id"].append(np.arange(count, dtype=np.int32))
                blocks["noise_level"].append(np.full(count, level, dtype=np.float32))
                blocks["noise_sigma"].append(np.full(count, sigma, dtype=np.float32))

            arrays = {k: np.concatenate(v, axis=0) for k, v in blocks.items()}
            # Deterministic shuffle within split/noise without mixing physical-state ownership.
            rng = np.random.default_rng(base_seed + {"train": 11, "val": 22, "test": 33}[split] * 9_999_991 + int(round(level * 1e9)))
            perm = rng.permutation(len(arrays["state_id"]))
            for k in list(arrays):
                arrays[k] = arrays[k][perm]
            arrays["q2"] = q2
            arrays["q2_observed"] = q2[obs_idx]
            arrays["observation_indices"] = obs_idx
            arrays["free_params"] = np.asarray(cfg["free_params"], dtype="U16")
            path = ndir / f"{split}.npz"
            _save_npz(path, arrays, compressed)
            actual_counts[str(level)][split] = int(len(arrays["state_id"]))
            print(f"[noise {level:g}] {split}: {len(rows)} states x {count} = {len(arrays['state_id'])} samples")

    update_metadata(out, {
        "status": "NOISY_DATA_READY",
        "noise": {
            **noise_cfg,
            "resolved_samples_per_state": per_state,
            "actual_sample_counts": actual_counts,
            "definition": "noise is added at the configured physical observation q2 points with sigma=noise_level*RMS(g_clean), then interpolated to model_input_points",
        },
    })
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, True if a.compressed else None)


if __name__ == "__main__":
    main()
