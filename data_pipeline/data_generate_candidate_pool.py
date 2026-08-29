from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_core import (
    PARAMETER_NAMES,
    compute_clean_forward,
    config_fingerprint,
    config_stage_name,
    default_output_dir,
    ensure_empty_dir,
    git_or_hash_version,
    load_config,
    sample_candidate_parameters,
    state_ids,
    update_metadata,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate clean physical-state candidate pool")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--candidate-count", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(config_path: str, output_dir: str | None = None, candidate_count: int | None = None,
        seed: int | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    ensure_empty_dir(out, overwrite=overwrite)

    params, sampling_meta = sample_candidate_parameters(cfg, count=candidate_count, seed=seed)
    ids = state_ids(config_stage_name(cfg), len(params))
    print(f"[candidate] computing clean forward bank for {len(params)} physical states")
    g_clean, q2, obs_idx = compute_clean_forward(params, cfg)

    rows = []
    for i in range(len(params)):
        row = {"state_id": str(ids[i]), "state_index": i}
        for j, name in enumerate(PARAMETER_NAMES):
            row[name] = float(params[i, j])
        rows.append(row)
    write_csv_rows(out / "candidate_states.csv", rows)
    np.savez_compressed(
        out / "candidate_g_clean.npz",
        state_id=ids,
        g_clean=g_clean,
        q2=np.asarray(q2, dtype=np.float32),
        observation_indices=np.asarray(obs_idx, dtype=np.int32),
    )

    update_metadata(out, {
        "stage": config_stage_name(cfg),
        "status": "CANDIDATE_POOL_READY",
        "free_params": list(cfg["free_params"]),
        "fixed_params": dict(cfg["fixed_params"]),
        "parameter_order": list(PARAMETER_NAMES),
        "parameter_ranges": dict(cfg["parameter_ranges"]),
        "sampling": sampling_meta,
        "candidate_state_count": int(len(params)),
        "forward": {
            **cfg["forward"],
            "q2_domain_locked": True,
            "observation_indices": [int(x) for x in obs_idx],
        },
        "identifiability": {
            "reference_noise": float(cfg["identifiability"]["reference_noise"]),
            "unacceptable_difference": dict(cfg["identifiability"]["unacceptable_difference"]),
        },
        "noise": dict(cfg.get("noise", {})),
        "split": dict(cfg.get("split", {})),
        "selection": dict(cfg.get("selection", {})),
        "config_path": str(Path(config_path).resolve()),
        "config_fingerprint": config_fingerprint(cfg),
        "code_version": git_or_hash_version(),
    })
    print(f"[done] {out / 'candidate_states.csv'}")
    print(f"[done] {out / 'candidate_g_clean.npz'}")
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.candidate_count, a.seed, a.overwrite)


if __name__ == "__main__":
    main()
