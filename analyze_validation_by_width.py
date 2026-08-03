from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_edges(text: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError("至少需要两个分桶边界")
    edges = np.asarray(values, dtype=np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("分桶边界必须严格递增")
    return edges


def stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 m*gamma 峰宽分组汇总 V2 验证指标。"
    )
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--width-edges", default="0,0.1,0.2,0.5,2.000001")
    parser.add_argument("--output-dir", default="./validation_results/width_groups")
    args = parser.parse_args()

    path = Path(args.metrics_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("CSV 没有数据")
    required = [
        "m_gamma", "f_relative_l2", "g_relative_l2_vs_clean",
        "resonance_relative_l2", "resonance_center_error",
        "resonance_height_relative_error", "visible_peak_eligible",
    ]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise RuntimeError(
            "CSV 缺少新版峰指标：%s。请先用修正版 validate_mc_transformer.py 重新验证。"
            % ", ".join(missing)
        )

    def col(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    width = col("m_gamma")
    f_rel = col("f_relative_l2")
    g_rel = col("g_relative_l2_vs_clean")
    res_rel = col("resonance_relative_l2")
    center = col("resonance_center_error")
    height = col("resonance_height_relative_error")
    eligible = np.asarray(
        [str(row["visible_peak_eligible"]).lower() in {"true", "1", "yes"} for row in rows]
    )
    edges = parse_edges(args.width_edges)

    records: List[Dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (width >= left) & (width < right)
        peak_mask = mask & eligible
        record: Dict[str, object] = {
            "width_left": float(left),
            "width_right": float(right),
            "sample_count": int(mask.sum()),
            "visible_peak_count": int(peak_mask.sum()),
            "f_relative_l2": stats(f_rel[mask]),
            "g_relative_l2_vs_clean": stats(g_rel[mask]),
            "resonance_relative_l2_visible": stats(res_rel[peak_mask]),
            "resonance_center_error_visible": stats(center[peak_mask]),
            "resonance_height_relative_error_visible": stats(height[peak_mask]),
        }
        records.append(record)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "width_group_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = output / "width_group_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "width_left", "width_right", "sample_count", "visible_peak_count",
            "f_mean", "f_median", "f_p90", "g_mean",
            "res_mean", "res_median", "res_p90",
            "center_mean", "height_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "width_left": record["width_left"],
                "width_right": record["width_right"],
                "sample_count": record["sample_count"],
                "visible_peak_count": record["visible_peak_count"],
                "f_mean": record["f_relative_l2"].get("mean", ""),
                "f_median": record["f_relative_l2"].get("median", ""),
                "f_p90": record["f_relative_l2"].get("p90", ""),
                "g_mean": record["g_relative_l2_vs_clean"].get("mean", ""),
                "res_mean": record["resonance_relative_l2_visible"].get("mean", ""),
                "res_median": record["resonance_relative_l2_visible"].get("median", ""),
                "res_p90": record["resonance_relative_l2_visible"].get("p90", ""),
                "center_mean": record["resonance_center_error_visible"].get("mean", ""),
                "height_mean": record["resonance_height_relative_error_visible"].get("mean", ""),
            })
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print("csv:", csv_path.resolve())


if __name__ == "__main__":
    main()
