from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_identifiability import run as run_ident
from analyze_parameter_coverage import run as run_coverage
from build_noisy_dataset import run as run_noise
from data_generate_candidate_pool import run as run_candidate
from pipeline_core import config_fingerprint, default_output_dir, load_config
from select_identifiable_states import run as run_select
from split_physical_states import run as run_split
from validate_dataset import run as run_validate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command data-only stage runner")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--alias-threshold", type=float)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--formal", action="store_true", help="override a config whose default_action is smoke")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _effective_config(config_path: str, smoke: bool, output_dir: str | None) -> tuple[Path, dict, Path]:
    cfg = load_config(config_path)
    if smoke:
        cfg = json.loads(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
        cfg["sampling"]["candidate_count"] = 240
        cfg["forward"]["model_input_points"] = 120
        cfg["forward"]["observation_points"] = 60
        cfg["forward"]["integration_points"] = 32
        cfg["identifiability"]["initial_neighbors"] = 32
        cfg["identifiability"]["max_neighbors"] = 240
        cfg["identifiability"]["query_batch_size"] = 32
        cfg["identifiability"]["hard_state_count"] = 40
        cfg["identifiability"]["continuous_profile"]["m_points"] = 11
        cfg["identifiability"]["continuous_profile"]["gamma_points"] = 41
        cfg["identifiability"]["continuous_profile"]["basis_build_chunk"] = 128
        cfg["identifiability"]["continuous_profile"]["truth_chunk"] = 16
        cfg["identifiability"]["continuous_profile"]["basis_profile_chunk"] = 256
        cfg["selection"]["min_separation_rms_snr"] = 0.0
        cfg["selection"]["max_parameter_cover_radius"] = 0.0
        cfg["selection"]["max_selected_states"] = 30
        cfg["coverage"]["projection_bins"] = 5
        cfg["split"]["min_val_states"] = 3
        cfg["split"]["min_test_states"] = 3
        cfg["split"]["max_train_cover_rms_snr"] = 1.0e12
        cfg["noise"]["noise_levels"] = [0.0, 0.002]
        for split in ("train", "val", "test"):
            cfg["noise"][f"samples_per_{split}_state"] = 2
            cfg["noise"][f"target_{split}_samples"] = None
        cfg["validation"]["direct_forward_sample"] = 8
        cfg["validation"]["min_test_parameter_span_fraction"] = 0.0
        base = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "data_pipeline_outputs" / f"stage_{cfg['stage'][0]}p_smoke"
    else:
        base = Path(output_dir) if output_dir else default_output_dir(cfg)
    base.parent.mkdir(parents=True, exist_ok=True)
    effective_dir = Path(__file__).resolve().parent.parent / "data_pipeline_outputs" / ".effective_configs"
    effective_dir.mkdir(parents=True, exist_ok=True)
    tag = f"stage_{cfg['stage'][0]}p_{'smoke' if smoke else 'formal'}"
    effective = effective_dir / f"{tag}.json"
    effective.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, ensure_ascii=False, indent=2), encoding="utf-8")
    return effective, cfg, base


def main() -> None:
    a = parse_args()
    raw_cfg = load_config(a.config)
    default_action = str(raw_cfg.get("default_action", "analysis")).lower()
    smoke = bool(a.smoke or (default_action == "smoke" and not a.formal and a.alias_threshold is None))
    effective_path, cfg, out = _effective_config(a.config, smoke, a.output_dir)
    alias_threshold = a.alias_threshold
    if smoke and alias_threshold is None:
        alias_threshold = 0.0

    print("=" * 100)
    print(f"Universal data pipeline stage : {cfg['stage']}")
    print(f"free_params                  : {cfg['free_params']}")
    print(f"fixed_params                 : {cfg['fixed_params']}")
    print(f"q2 domain                    : [{cfg['forward']['q2_min']}, {cfg['forward']['q2_max']}] LOCKED")
    print(f"mode                         : {'SMOKE' if smoke else 'FORMAL/ANALYSIS'}")
    print(f"output                       : {out}")
    print("NO network training is performed by this pipeline.")
    print("=" * 100)

    if a.overwrite or not (out / "candidate_states.csv").exists():
        run_candidate(str(effective_path), str(out), overwrite=a.overwrite)
    else:
        meta_path = out / "metadata.json"
        if not meta_path.exists():
            raise RuntimeError("candidate output exists without metadata.json; use --overwrite")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("config_fingerprint") != config_fingerprint(cfg):
            raise RuntimeError("existing output was built from a different config; use --overwrite")
        print("[reuse] candidate pool")
    if a.overwrite or not (out / "identifiability_results.csv").exists():
        run_ident(str(effective_path), str(out), backend="auto")
    else:
        print("[reuse] identifiability results")

    if a.analysis_only or alias_threshold is None:
        print("=" * 100)
        print("Candidate pool + identifiability analysis complete.")
        print("No final alias cutoff was supplied, so selection/split/noise are intentionally NOT frozen.")
        print(f"Review: {out / 'alias_summary.csv'}")
        print(f"Review: {out / 'hard_states.csv'}")
        print("=" * 100)
        return

    run_select(
        str(effective_path), str(out), float(alias_threshold),
        smoke_permissive=smoke,
    )
    run_coverage(str(effective_path), str(out))
    run_split(str(effective_path), str(out), smoke_permissive=smoke)
    run_noise(str(effective_path), str(out))
    run_validate(str(effective_path), str(out), smoke_permissive=smoke)
    print("=" * 100)
    print(f"{cfg['stage']} full data-only pipeline complete and validated.")
    if smoke:
        print("This output is SMOKE ONLY and must not be released for training.")
    else:
        print("This output is eligible for data review; network training remains a separate responsibility.")
    print("=" * 100)


if __name__ == "__main__":
    main()
