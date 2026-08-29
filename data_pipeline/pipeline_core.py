from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mc_pool_config import DEFAULT_PHYSICS, PARAMETER_NAMES, atomic_write_json
from mc_physics import scaled_forward_observation_numpy, valid_parameter_mask_numpy

PARAM_INDEX = {name: i for i, name in enumerate(PARAMETER_NAMES)}
LINEAR_PARAMS = ("a1", "a2", "a3")
NONLINEAR_PARAMS = ("m", "gamma")


def canonical_indices(master_points: int, observation_points: int) -> np.ndarray:
    """Deterministic endpoint-preserving observation indices; self-contained data-pipeline copy."""
    master_points = int(master_points)
    observation_points = int(observation_points)
    if not (2 <= observation_points <= master_points):
        raise ValueError("require 2 <= observation_points <= model_input_points")
    if observation_points == master_points:
        return np.arange(master_points, dtype=np.int64)

    def rounded_linspace(total: int, count: int) -> np.ndarray:
        idx = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
        idx[0] = 0
        idx[-1] = total - 1
        if len(np.unique(idx)) != count:
            idx = np.floor(np.linspace(0, total, count, endpoint=False)).astype(np.int64)
            idx[-1] = total - 1
            idx = np.unique(idx)
            if len(idx) != count:
                raise RuntimeError("could not construct unique observation indices")
        return idx

    if master_points == 1000 and observation_points == 500:
        return rounded_linspace(master_points, 500)
    if master_points == 1000 and observation_points == 100:
        parent = rounded_linspace(master_points, 500)
        sub = rounded_linspace(len(parent), 100)
        idx = parent[sub]
        if len(np.unique(idx)) != 100:
            raise RuntimeError("nested q100 index construction failed")
        return idx.astype(np.int64)
    return rounded_linspace(master_points, observation_points)


def interpolation_plan(q_obs: np.ndarray, q_model: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_obs = np.asarray(q_obs, dtype=np.float64)
    q_model = np.asarray(q_model, dtype=np.float64)
    if np.any(np.diff(q_obs) <= 0) or np.any(np.diff(q_model) <= 0):
        raise ValueError("q grids must be strictly increasing")
    if q_obs[0] > q_model[0] + 1e-12 or q_obs[-1] < q_model[-1] - 1e-12:
        raise ValueError("observation grid must cover model grid endpoints")
    hi = np.searchsorted(q_obs, q_model, side="left")
    hi = np.clip(hi, 1, len(q_obs) - 1)
    lo = hi - 1
    x0 = q_obs[lo]
    x1 = q_obs[hi]
    w = (q_model - x0) / np.maximum(x1 - x0, 1e-30)
    return lo.astype(np.int64), hi.astype(np.int64), w.astype(np.float32)


def interpolate_rows(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, w: np.ndarray, chunk_rows: int = 1024) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty((values.shape[0], len(w)), dtype=np.float32)
    ww = np.asarray(w, dtype=np.float32)[None, :]
    for start in range(0, len(values), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(values))
        block = values[start:stop]
        left = block[:, lo]
        right = block[:, hi]
        out[start:stop] = left + (right - left) * ww
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    free = list(cfg.get("free_params", []))
    if not free or len(set(free)) != len(free):
        raise ValueError("config.free_params must be a non-empty unique list")
    unknown = sorted(set(free) - set(PARAMETER_NAMES))
    if unknown:
        raise ValueError(f"unknown free parameters: {unknown}")
    fixed = dict(cfg.get("fixed_params", {}))
    if set(free) & set(fixed):
        raise ValueError("a parameter cannot be both free and fixed")
    if set(free) | set(fixed) != set(PARAMETER_NAMES):
        missing = sorted(set(PARAMETER_NAMES) - (set(free) | set(fixed)))
        raise ValueError(f"every parameter must be free or fixed; missing={missing}")
    ranges = cfg.get("parameter_ranges", {})
    for name in PARAMETER_NAMES:
        if name in free:
            if name not in ranges:
                raise ValueError(f"missing parameter range for free parameter {name}")
            lo, hi = [float(x) for x in ranges[name]]
            if not hi > lo:
                raise ValueError(f"invalid parameter range for {name}: {ranges[name]}")
        else:
            if not np.isfinite(float(fixed[name])):
                raise ValueError(f"fixed parameter {name} is non-finite")
    forward = cfg.get("forward", {})
    q2_min = float(forward.get("q2_min", DEFAULT_PHYSICS.q2_min))
    q2_max = float(forward.get("q2_max", DEFAULT_PHYSICS.q2_max))
    if not np.isclose(q2_min, DEFAULT_PHYSICS.q2_min, atol=0, rtol=0):
        raise ValueError("q2_min differs from the project's fixed physics domain")
    if not np.isclose(q2_max, DEFAULT_PHYSICS.q2_max, atol=0, rtol=0):
        raise ValueError("q2_max differs from the project's fixed physics domain")
    model_pts = int(forward.get("model_input_points", DEFAULT_PHYSICS.q2_points))
    obs_pts = int(forward.get("observation_points", model_pts))
    if not (2 <= obs_pts <= model_pts):
        raise ValueError("require 2 <= observation_points <= model_input_points")
    ident = cfg.get("identifiability", {})
    if float(ident.get("reference_noise", 0.0)) <= 0:
        raise ValueError("identifiability.reference_noise must be positive")
    ud = ident.get("unacceptable_difference", {})
    for name in free:
        key = "gamma_factor" if name == "gamma" else f"{name}_abs"
        if key not in ud:
            raise ValueError(f"missing unacceptable_difference.{key}")
        if float(ud[key]) <= (1.0 if key == "gamma_factor" else 0.0):
            raise ValueError(f"invalid unacceptable difference for {name}")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_stage_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("stage", "stage")).strip()


def default_output_dir(cfg: dict[str, Any]) -> Path:
    return project_root() / str(cfg.get("output_dir", f"data_pipeline_outputs/{config_stage_name(cfg).lower()}"))


def ensure_empty_dir(path: str | Path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    names.append(key)
                    seen.add(key)
        fieldnames = names
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: jsonable(row.get(k, "")) for k in fieldnames})


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value



def config_fingerprint(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def git_or_hash_version() -> str:
    root = project_root()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=3
        ).strip()
        if out:
            return f"git:{out}"
    except Exception:
        pass
    h = hashlib.sha256()
    for path in sorted((root / "data_pipeline").rglob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return f"pipeline-sha256:{h.hexdigest()[:16]}"


def update_metadata(output_dir: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    path = Path(output_dir) / "metadata.json"
    meta: dict[str, Any] = {}
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
    deep_update(meta, patch)
    atomic_write_json(path, meta)
    return meta


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value


def make_physics(cfg: dict[str, Any], model_input_points: int | None = None):
    fwd = cfg["forward"]
    qpts = int(model_input_points or fwd["model_input_points"])
    return replace(DEFAULT_PHYSICS, q2_points=qpts)


def parameter_matrix_from_rows(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    p = np.empty((len(rows), 5), dtype=np.float64)
    for i, name in enumerate(PARAMETER_NAMES):
        p[:, i] = [float(r[name]) for r in rows]
    return p


def normalized_parameter_matrix(params: np.ndarray, cfg: dict[str, Any], names: Sequence[str] | None = None) -> np.ndarray:
    names = list(names or cfg["free_params"])
    cols = []
    for name in names:
        x = np.asarray(params[:, PARAM_INDEX[name]], dtype=np.float64)
        if name == "gamma":
            x = np.log10(np.clip(x, 1e-300, None))
            lo, hi = cfg["parameter_ranges"][name]
            lo, hi = math.log10(float(lo)), math.log10(float(hi))
        else:
            lo, hi = [float(v) for v in cfg["parameter_ranges"][name]]
        cols.append((x - lo) / max(hi - lo, 1e-30))
    return np.column_stack(cols) if cols else np.zeros((len(params), 0), dtype=np.float64)


def parameter_pair_metrics(a: np.ndarray, b: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    free = cfg["free_params"]
    za = normalized_parameter_matrix(np.asarray(a, dtype=np.float64).reshape(1, 5), cfg, free)[0]
    zb = normalized_parameter_matrix(np.asarray(b, dtype=np.float64).reshape(1, 5), cfg, free)[0]
    out: dict[str, float] = {}
    raw_terms = []
    for name in PARAMETER_NAMES:
        d = float(b[PARAM_INDEX[name]] - a[PARAM_INDEX[name]])
        out[f"delta_{name}"] = d
        out[f"abs_delta_{name}"] = abs(d)
        if name in free:
            raw_terms.append(d * d)
    ga, gb = float(a[4]), float(b[4])
    out["gamma_factor"] = float(max(ga / max(gb, 1e-300), gb / max(ga, 1e-300)))
    out["raw_parameter_distance"] = float(math.sqrt(sum(raw_terms)))
    out["normalized_parameter_distance"] = float(np.linalg.norm(zb - za))
    return out


def unacceptable_mask(truth: np.ndarray, candidates: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    ud = cfg["identifiability"]["unacceptable_difference"]
    mask = np.zeros(len(candidates), dtype=bool)
    for name in cfg["free_params"]:
        j = PARAM_INDEX[name]
        if name == "gamma":
            t = max(float(truth[j]), 1e-300)
            c = np.maximum(np.asarray(candidates[:, j], dtype=np.float64), 1e-300)
            factor = np.maximum(c / t, t / c)
            mask |= factor >= float(ud["gamma_factor"])
        else:
            mask |= np.abs(np.asarray(candidates[:, j], dtype=np.float64) - float(truth[j])) >= float(ud[f"{name}_abs"])
    return mask


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    u = np.empty((n, dim), dtype=np.float64)
    for j in range(dim):
        x = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(n)
        rng.shuffle(x)
        u[:, j] = x
    return u


def sample_candidate_parameters(cfg: dict[str, Any], count: int | None = None, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    sampling = cfg["sampling"]
    n = int(count or sampling["candidate_count"])
    if n < 8:
        raise ValueError("candidate_count must be at least 8")
    seed = int(seed if seed is not None else sampling["seed"])
    rng = np.random.default_rng(seed)
    free = list(cfg["free_params"])
    method = str(sampling.get("method", "latin_hypercube")).lower()
    if method not in {"latin_hypercube", "random", "uniform_grid"}:
        raise ValueError(f"unsupported sampling method: {method}")

    if method == "uniform_grid":
        d = len(free)
        per = max(2, int(math.ceil(n ** (1.0 / max(d, 1)))))
        axes = []
        for name in free:
            lo, hi = [float(x) for x in cfg["parameter_ranges"][name]]
            if name == "gamma" and str(sampling.get("gamma_scale", "log10")) == "log10":
                axes.append(np.geomspace(lo, hi, per, dtype=np.float64))
            else:
                axes.append(np.linspace(lo, hi, per, dtype=np.float64))
        points = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
        if len(points) > n:
            idx = np.linspace(0, len(points) - 1, n, dtype=np.int64)
            points = points[idx]
        base = points
    else:
        u = latin_hypercube(n, len(free), rng) if method == "latin_hypercube" else rng.random((n, len(free)))
        cols = []
        for j, name in enumerate(free):
            lo, hi = [float(x) for x in cfg["parameter_ranges"][name]]
            if name == "gamma" and str(sampling.get("gamma_scale", "log10")) == "log10":
                x = 10.0 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * u[:, j])
            else:
                x = lo + (hi - lo) * u[:, j]
            cols.append(x)
        base = np.column_stack(cols)

    params = np.empty((len(base), 5), dtype=np.float64)
    for name in PARAMETER_NAMES:
        if name in free:
            params[:, PARAM_INDEX[name]] = base[:, free.index(name)]
        else:
            params[:, PARAM_INDEX[name]] = float(cfg["fixed_params"][name])

    # Exact corners are useful for coverage diagnostics. Insert only physically valid corners.
    if method != "uniform_grid" and bool(sampling.get("include_corners", True)):
        d = len(free)
        max_corners = int(sampling.get("max_corners", 32))
        corner_bits = list(itertools.product([0, 1], repeat=d))[:max_corners]
        corners = []
        for bits in corner_bits:
            p = np.empty(5, dtype=np.float64)
            for name in PARAMETER_NAMES:
                if name in free:
                    lo, hi = [float(v) for v in cfg["parameter_ranges"][name]]
                    p[PARAM_INDEX[name]] = (lo, hi)[bits[free.index(name)]]
                else:
                    p[PARAM_INDEX[name]] = float(cfg["fixed_params"][name])
            corners.append(p)
        corners = np.asarray(corners, dtype=np.float64)
        valid = valid_parameter_mask_numpy(corners, config=DEFAULT_PHYSICS)
        corners = corners[valid]
        take = min(len(corners), len(params))
        if take:
            params[:take] = corners[:take]

    valid = valid_parameter_mask_numpy(params, config=DEFAULT_PHYSICS)
    accepted = params[valid]
    attempts = len(params)
    # Rejection refill for 5P physicality.
    max_rounds = int(sampling.get("physical_refill_rounds", 20))
    round_id = 0
    while len(accepted) < n and round_id < max_rounds:
        need = n - len(accepted)
        round_id += 1
        refill_cfg = dict(cfg)
        refill_sampling = dict(cfg["sampling"])
        refill_sampling["include_corners"] = False
        refill_sampling["candidate_count"] = max(need * 2, 32)
        refill_cfg["sampling"] = refill_sampling
        # Generate directly to avoid recursive corner logic.
        u = latin_hypercube(refill_sampling["candidate_count"], len(free), rng)
        p2 = np.empty((len(u), 5), dtype=np.float64)
        for name in PARAMETER_NAMES:
            if name in free:
                j = free.index(name)
                lo, hi = [float(x) for x in cfg["parameter_ranges"][name]]
                if name == "gamma" and str(sampling.get("gamma_scale", "log10")) == "log10":
                    p2[:, PARAM_INDEX[name]] = 10.0 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * u[:, j])
                else:
                    p2[:, PARAM_INDEX[name]] = lo + (hi - lo) * u[:, j]
            else:
                p2[:, PARAM_INDEX[name]] = float(cfg["fixed_params"][name])
        attempts += len(p2)
        p2 = p2[valid_parameter_mask_numpy(p2, config=DEFAULT_PHYSICS)]
        if len(p2):
            accepted = np.vstack([accepted, p2])
    if len(accepted) < n:
        raise RuntimeError(f"physical rejection sampling produced only {len(accepted)} / {n} requested states")
    accepted = accepted[:n]
    return accepted, {
        "method": method,
        "seed": seed,
        "requested_count": n,
        "raw_draw_count": attempts,
        "physical_acceptance_rate": float(n / attempts),
        "gamma_scale": sampling.get("gamma_scale", "linear"),
    }


def compute_clean_forward(params: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fwd = cfg["forward"]
    physics = make_physics(cfg)
    g = scaled_forward_observation_numpy(
        params,
        integration_points=int(fwd.get("integration_points", 128)),
        config=physics,
    ).astype(np.float32)
    q2 = np.linspace(physics.q2_min, physics.q2_max, physics.q2_points, dtype=np.float64)
    obs_idx = canonical_indices(int(physics.q2_points), int(fwd["observation_points"]))
    return g, q2, np.asarray(obs_idx, dtype=np.int32)


def state_ids(stage: str, n: int) -> np.ndarray:
    width = max(8, len(str(max(n - 1, 0))))
    return np.asarray([f"{stage}_S{i:0{width}d}" for i in range(n)], dtype=f"U{len(stage)+width+2}")


def master_rms(g: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(g, dtype=np.float64) ** 2, axis=1))


def symmetric_rms_snr(g_a: np.ndarray, rms_a: np.ndarray, g_b: np.ndarray, rms_b: np.ndarray, reference_noise: float) -> np.ndarray:
    d = np.asarray(g_a, dtype=np.float64) - np.asarray(g_b, dtype=np.float64)
    rms_d = np.sqrt(np.mean(d * d, axis=-1))
    sigma = float(reference_noise) * 0.5 * (np.asarray(rms_a, dtype=np.float64) + np.asarray(rms_b, dtype=np.float64))
    return rms_d / np.maximum(sigma, 1e-30)


def truth_rms_snr(g_true: np.ndarray, g_alt: np.ndarray, rms_true: float, reference_noise: float) -> tuple[float, float]:
    d = np.asarray(g_true, dtype=np.float64) - np.asarray(g_alt, dtype=np.float64)
    gdist = float(np.sqrt(np.mean(d * d)))
    score = gdist / max(float(reference_noise) * float(rms_true), 1e-30)
    return gdist, score


def _rademacher_projection(x: np.ndarray, dims: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(x.shape[1], int(dims)))
    signs /= np.float32(math.sqrt(max(int(dims), 1)))
    return (x @ signs).astype(np.float32)


def find_candidate_aliases(
    *, params: np.ndarray, g_master: np.ndarray, obs_idx: np.ndarray, state_id: np.ndarray,
    cfg: dict[str, Any], backend: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ident = cfg["identifiability"]
    reference_noise = float(ident["reference_noise"])
    n = len(params)
    obs = np.asarray(g_master[:, np.asarray(obs_idx, dtype=np.int64)], dtype=np.float32)
    rms = master_rms(g_master)
    exact_limit = int(ident.get("exact_kdtree_limit", 50000))
    backend = str(backend or ident.get("search_backend", "auto")).lower()
    if backend == "auto":
        backend = "exact_kdtree" if n <= exact_limit else "projected_kdtree"
    if backend not in {"exact_kdtree", "projected_kdtree"}:
        raise ValueError("search backend must be auto, exact_kdtree, or projected_kdtree")
    if backend == "projected_kdtree":
        feat = _rademacher_projection(obs, int(ident.get("projection_dims", 24)), int(ident.get("search_seed", 20260880)))
        exact_search = False
    else:
        feat = obs
        exact_search = True
    tree = cKDTree(np.asarray(feat, dtype=np.float64), compact_nodes=True, balanced_tree=True)
    max_neighbors = min(n, int(ident.get("max_neighbors", 4096)))
    initial_neighbors = min(max_neighbors, max(8, int(ident.get("initial_neighbors", 128))))
    query_batch = max(1, int(ident.get("query_batch_size", 64)))
    rows: list[dict[str, Any]] = []
    unresolved = 0
    exact_fallbacks = 0

    for start in range(0, n, query_batch):
        stop = min(start + query_batch, n)
        for i in range(start, stop):
            k = initial_neighbors
            chosen: np.ndarray | None = None
            while True:
                _, nn = tree.query(np.asarray(feat[i], dtype=np.float64), k=k, workers=-1)
                nn = np.atleast_1d(nn).astype(np.int64)
                nn = nn[nn != i]
                remote = unacceptable_mask(params[i], params[nn], cfg)
                if np.any(remote):
                    chosen = nn[remote]
                    break
                if k >= max_neighbors:
                    break
                k = min(max_neighbors, max(k + 1, k * 2))
            if chosen is None and exact_search and max_neighbors < n:
                # This is an exact no-NxN-memory fallback for the rare truth whose first K g-neighbors
                # are all parameter-near. Querying one row against the whole tree is memory-bounded.
                exact_fallbacks += 1
                _, nn = tree.query(np.asarray(feat[i], dtype=np.float64), k=n, workers=-1)
                nn = np.atleast_1d(nn).astype(np.int64)
                nn = nn[nn != i]
                remote = unacceptable_mask(params[i], params[nn], cfg)
                if np.any(remote):
                    chosen = nn[remote]
            if chosen is None or len(chosen) == 0:
                unresolved += 1
                row = {
                    "state_id": str(state_id[i]),
                    "state_index": i,
                    "nearest_alias_state_id": "",
                    "nearest_alias_state_index": -1,
                    "alias_score": float("inf"),
                    "nearest_alias_rms_snr": float("inf"),
                    "g_rms_distance": float("inf"),
                    "search_resolved": False,
                    "search_exact": exact_search,
                }
                for name in PARAMETER_NAMES:
                    row[name] = float(params[i, PARAM_INDEX[name]])
                    row[f"alias_{name}"] = float("nan")
                    row[f"delta_{name}"] = float("nan")
                    row[f"abs_delta_{name}"] = float("nan")
                row["gamma_factor"] = float("nan")
                row["raw_parameter_distance"] = float("nan")
                row["normalized_parameter_distance"] = float("nan")
                rows.append(row)
                continue

            # For exact_kdtree the chosen list is ordered by true g Euclidean distance, but recompute
            # exact distances anyway so projected search and exact search share one code path.
            diff = np.asarray(obs[chosen], dtype=np.float64) - np.asarray(obs[i], dtype=np.float64)[None, :]
            gdist = np.sqrt(np.mean(diff * diff, axis=1))
            j = int(chosen[int(np.argmin(gdist))])
            exact_gdist, score = truth_rms_snr(obs[i], obs[j], rms[i], reference_noise)
            pm = parameter_pair_metrics(params[i], params[j], cfg)
            row = {
                "state_id": str(state_id[i]),
                "state_index": i,
                "nearest_alias_state_id": str(state_id[j]),
                "nearest_alias_state_index": j,
                "alias_score": score,
                "nearest_alias_rms_snr": score,
                "g_rms_distance": exact_gdist,
                "noise_sigma_reference": float(reference_noise * rms[i]),
                "search_resolved": True,
                "search_exact": exact_search,
                "neighbors_examined_limit": int(max_neighbors),
            }
            for name in PARAMETER_NAMES:
                row[name] = float(params[i, PARAM_INDEX[name]])
                row[f"alias_{name}"] = float(params[j, PARAM_INDEX[name]])
            row.update(pm)
            rows.append(row)
        print(f"[alias] {stop} / {n}")

    return rows, {
        "backend": backend,
        "search_exact": exact_search,
        "candidate_count": n,
        "observation_dimension": int(obs.shape[1]),
        "max_neighbors": int(max_neighbors),
        "initial_neighbors": int(initial_neighbors),
        "unresolved_count": int(unresolved),
        "exact_full_tree_fallback_count": int(exact_fallbacks),
        "projection_dims": int(ident.get("projection_dims", 24)) if not exact_search else None,
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {k: float("nan") for k in ("min", "p05", "p25", "p50", "p75", "p95", "max", "mean")}
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def pairwise_min_snr_between(
    g_left: np.ndarray, rms_left: np.ndarray, g_right: np.ndarray, rms_right: np.ndarray,
    reference_noise: float, block: int = 256,
) -> dict[str, float]:
    if len(g_left) == 0 or len(g_right) == 0:
        return {"min_rms_snr": float("nan"), "median_nearest_rms_snr": float("nan"), "max_nearest_rms_snr": float("nan")}
    nearest = np.full(len(g_left), np.inf, dtype=np.float64)
    right = np.asarray(g_right, dtype=np.float64)
    rnorm = np.sum(right * right, axis=1)
    d = float(g_left.shape[1])
    for s in range(0, len(g_left), block):
        e = min(s + block, len(g_left))
        a = np.asarray(g_left[s:e], dtype=np.float64)
        anorm = np.sum(a * a, axis=1)
        dist2 = np.maximum(anorm[:, None] + rnorm[None, :] - 2.0 * (a @ right.T), 0.0)
        rmsd = np.sqrt(dist2 / d)
        sigma = float(reference_noise) * 0.5 * (np.asarray(rms_left[s:e])[:, None] + np.asarray(rms_right)[None, :])
        snr = rmsd / np.maximum(sigma, 1e-30)
        nearest[s:e] = np.min(snr, axis=1)
    return {
        "min_rms_snr": float(np.min(nearest)),
        "median_nearest_rms_snr": float(np.median(nearest)),
        "max_nearest_rms_snr": float(np.max(nearest)),
    }


def nearest_to_train_scores(g: np.ndarray, rms: np.ndarray, train_idx: np.ndarray, query_idx: np.ndarray, reference_noise: float) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_idx, dtype=np.int64)
    query = np.asarray(query_idx, dtype=np.int64)
    scores = np.full(len(query), np.inf, dtype=np.float64)
    neighbor = np.full(len(query), -1, dtype=np.int64)
    if len(query) == 0 or len(train) == 0:
        return scores, neighbor
    gt = np.asarray(g[train], dtype=np.float64)
    tnorm = np.sum(gt * gt, axis=1)
    d = float(g.shape[1])
    for q0 in range(0, len(query), 256):
        q1 = min(q0 + 256, len(query))
        ids = query[q0:q1]
        a = np.asarray(g[ids], dtype=np.float64)
        anorm = np.sum(a * a, axis=1)
        dist2 = np.maximum(anorm[:, None] + tnorm[None, :] - 2.0 * (a @ gt.T), 0.0)
        rmsd = np.sqrt(dist2 / d)
        sigma = float(reference_noise) * 0.5 * (rms[ids, None] + rms[train][None, :])
        snr = rmsd / np.maximum(sigma, 1e-30)
        k = np.argmin(snr, axis=1)
        scores[q0:q1] = snr[np.arange(len(ids)), k]
        neighbor[q0:q1] = train[k]
    return scores, neighbor


def noise_dir_name(level: float) -> str:
    pct = float(level) * 100.0
    text = (f"{pct:.8g}").replace("-", "m").replace(".", "p")
    return f"noise_{text}pct"


def resolve_per_state_count(noise_cfg: dict[str, Any], split: str, n_states: int) -> int:
    explicit = noise_cfg.get(f"samples_per_{split}_state")
    if explicit is not None:
        return max(1, int(explicit))
    target = noise_cfg.get(f"target_{split}_samples")
    if target is None:
        return 1
    return max(1, int(math.ceil(float(target) / max(int(n_states), 1))))


def add_noise_and_interpolate(
    clean_master: np.ndarray, q_master: np.ndarray, obs_idx: np.ndarray,
    noise_level: float, count: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    clean_master = np.asarray(clean_master, dtype=np.float32)
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    clean_obs = clean_master[obs_idx]
    rms = float(np.sqrt(np.mean(clean_master.astype(np.float64) ** 2)))
    sigma = float(noise_level) * rms
    rng = np.random.default_rng(int(seed))
    if noise_level > 0:
        noisy_obs = clean_obs[None, :] + np.float32(sigma) * rng.standard_normal((count, len(obs_idx))).astype(np.float32)
    else:
        noisy_obs = np.repeat(clean_obs[None, :], count, axis=0)
    q_obs = np.asarray(q_master[obs_idx], dtype=np.float64)
    lo, hi, w = interpolation_plan(q_obs, np.asarray(q_master, dtype=np.float64))
    noisy_master = interpolate_rows(noisy_obs, lo, hi, w).astype(np.float32)
    return noisy_master, noisy_obs.astype(np.float32), sigma


def stable_state_seed(base_seed: int, split: str, state_index: int, noise_level: float) -> int:
    split_code = {"train": 11, "val": 22, "test": 33}[split]
    return int(base_seed) + split_code * 1_000_003 + (int(state_index) + 1) * 100_003 + int(round(float(noise_level) * 1e9))

# ------------------------- continuous profiled alias analysis -------------------------
# This keeps the proven Exp35-37 idea inside the new config-driven core: (m,gamma) are
# nonlinear bank coordinates, while whichever of a1/a2/a3 are free are profiled jointly.


def _qp_objective(x: np.ndarray, G: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("tbi,bij,tbj->tb", x, G, x, optimize=True) - 2.0 * np.sum(b * x, axis=-1)


def _solve_box_qp_1d(G: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    den = np.maximum(G[None, :, 0, 0], 1e-30)
    x = np.clip(b[..., 0] / den, lo[..., 0], hi[..., 0])[..., None]
    return x, _qp_objective(x, G, b)


def _solve_box_qp_2d(G: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, B, _ = b.shape
    best = np.full((T, B), np.inf, dtype=np.float64)
    best_x = np.zeros((T, B, 2), dtype=np.float64)
    det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
    safe = np.abs(det) > 1e-24
    denom = np.where(safe, det, 1.0)[None, :]
    x0 = np.empty((T, B, 2), dtype=np.float64)
    x0[..., 0] = (b[..., 0] * G[None, :, 1, 1] - b[..., 1] * G[None, :, 0, 1]) / denom
    x0[..., 1] = (G[None, :, 0, 0] * b[..., 1] - G[None, :, 1, 0] * b[..., 0]) / denom
    valid = safe[None, :] & np.all((x0 >= lo) & (x0 <= hi), axis=-1)
    val = _qp_objective(x0, G, b)
    take = valid & (val < best)
    best[take] = val[take]
    best_x[take] = x0[take]
    for fixed_dim in (0, 1):
        free_dim = 1 - fixed_dim
        for side in (0, 1):
            xf = lo[..., fixed_dim] if side == 0 else hi[..., fixed_dim]
            x = np.empty((T, B, 2), dtype=np.float64)
            x[..., fixed_dim] = xf
            num = b[..., free_dim] - G[None, :, free_dim, fixed_dim] * xf
            den = np.maximum(G[None, :, free_dim, free_dim], 1e-30)
            x[..., free_dim] = np.clip(num / den, lo[..., free_dim], hi[..., free_dim])
            val = _qp_objective(x, G, b)
            take = val < best
            best[take] = val[take]
            best_x[take] = x[take]
    return best_x, best


def _solve_box_qp_3d(G: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, B, _ = b.shape
    best = np.full((T, B), np.inf, dtype=np.float64)
    best_x = np.zeros((T, B, 3), dtype=np.float64)
    inv = np.linalg.pinv(G)
    x0 = np.einsum("bij,tbj->tbi", inv, b, optimize=True)
    valid = np.all((x0 >= lo) & (x0 <= hi), axis=-1)
    val = _qp_objective(x0, G, b)
    take = valid & (val < best)
    best[take] = val[take]
    best_x[take] = x0[take]
    for fixed_dim in range(3):
        free = [k for k in range(3) if k != fixed_dim]
        Gff = G[:, free][:, :, free]
        Gfa = G[:, free, fixed_dim]
        for side in (0, 1):
            xf = lo[..., fixed_dim] if side == 0 else hi[..., fixed_dim]
            bff = b[..., free] - xf[..., None] * Gfa[None, :, :]
            xfree, _ = _solve_box_qp_2d(Gff, bff, lo[..., free], hi[..., free])
            x = np.empty((T, B, 3), dtype=np.float64)
            x[..., fixed_dim] = xf
            x[..., free] = xfree
            val = _qp_objective(x, G, b)
            take = val < best
            best[take] = val[take]
            best_x[take] = x[take]
    return best_x, best


def _solve_box_qp(G: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = b.shape[-1]
    if d == 1:
        return _solve_box_qp_1d(G, b, lo, hi)
    if d == 2:
        return _solve_box_qp_2d(G, b, lo, hi)
    if d == 3:
        return _solve_box_qp_3d(G, b, lo, hi)
    raise ValueError("continuous linear profiling supports 1-3 free linear coefficients")


def _boundary_qp(G: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 fixed_dim: int, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = b.shape[-1]
    T, B, _ = b.shape
    x = np.empty((T, B, d), dtype=np.float64)
    x[..., fixed_dim] = value
    free = [k for k in range(d) if k != fixed_dim]
    if not free:
        return x, _qp_objective(x, G, b)
    Gff = G[:, free][:, :, free]
    Gfa = G[:, free, fixed_dim]
    bff = b[..., free] - value[..., None] * Gfa[None, :, :]
    if len(free) == 1:
        den = np.maximum(Gff[None, :, 0, 0], 1e-30)
        xf = np.clip(bff[..., 0] / den, lo[..., free[0]], hi[..., free[0]])
        x[..., free[0]] = xf
    else:
        xf, _ = _solve_box_qp_2d(Gff, bff, lo[..., free], hi[..., free])
        x[..., free] = xf
    return x, _qp_objective(x, G, b)


def _remote_profile_qp(G: np.ndarray, b: np.ndarray, truth_x: np.ndarray,
                       lo_vec: np.ndarray, hi_vec: np.ndarray, remote_delta: np.ndarray,
                       nonlinear_remote: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, B, d = b.shape
    lo = np.broadcast_to(lo_vec.reshape(1, 1, d), (T, B, d))
    hi = np.broadcast_to(hi_vec.reshape(1, 1, d), (T, B, d))
    x0, obj0 = _solve_box_qp(G, b, lo, hi)
    coeff_remote = np.any(np.abs(x0 - truth_x[:, None, :]) >= remote_delta[None, None, :], axis=-1)
    valid0 = nonlinear_remote | coeff_remote
    best_obj = np.where(valid0, obj0, np.inf)
    best_x = x0.copy()
    need = ~valid0
    if np.any(need):
        for k in range(d):
            for sign in (-1.0, 1.0):
                boundary = truth_x[:, k] + sign * remote_delta[k]
                valid_truth_side = (boundary >= lo_vec[k]) & (boundary <= hi_vec[k])
                if not np.any(valid_truth_side):
                    continue
                value = np.broadcast_to(boundary[:, None], (T, B))
                xb, objb = _boundary_qp(G, b, lo, hi, k, value)
                valid = need & valid_truth_side[:, None]
                take = valid & (objb < best_obj)
                best_obj[take] = objb[take]
                best_x[take] = xb[take]
    return best_x, best_obj


def _unit_linear_component(name: str, cfg: dict[str, Any], obs_idx: np.ndarray) -> np.ndarray:
    physics = make_physics(cfg)
    p = np.asarray([[0.0, 0.0, 0.0, 0.8, 0.2]], dtype=np.float64)
    ref = 0.2 if name == "a1" else 0.05
    p[0, PARAM_INDEX[name]] = ref
    g = scaled_forward_observation_numpy(
        p, integration_points=int(cfg["forward"].get("integration_points", 128)), config=physics
    )[0].astype(np.float64)
    return g[np.asarray(obs_idx, dtype=np.int64)] / ref


def _continuous_basis_grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cp = cfg["identifiability"].get("continuous_profile", {})
    if "m" in cfg["free_params"]:
        mlo, mhi = [float(x) for x in cfg["parameter_ranges"]["m"]]
        mgrid = np.linspace(mlo, mhi, int(cp.get("m_points", 51)), dtype=np.float64)
    else:
        mgrid = np.asarray([float(cfg["fixed_params"]["m"])], dtype=np.float64)
    if "gamma" in cfg["free_params"]:
        glo, ghi = [float(x) for x in cfg["parameter_ranges"]["gamma"]]
        ggrid = np.geomspace(glo, ghi, int(cp.get("gamma_points", 401)), dtype=np.float64)
    else:
        ggrid = np.asarray([float(cfg["fixed_params"]["gamma"])], dtype=np.float64)
    return np.repeat(mgrid, len(ggrid)), np.tile(ggrid, len(mgrid))


def continuous_profile_aliases(*, params: np.ndarray, g_master: np.ndarray, obs_idx: np.ndarray,
                               state_id: np.ndarray, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cp = cfg["identifiability"].get("continuous_profile", {})
    if not bool(cp.get("enabled", False)):
        return [], {"enabled": False}
    free_linear = [x for x in LINEAR_PARAMS if x in cfg["free_params"]]
    if len(free_linear) == 0:
        return [], {"enabled": False, "reason": "no free linear coefficients"}
    if ("m" in cfg["free_params"] or "gamma" in cfg["free_params"]) and "a1" not in free_linear:
        return [], {"enabled": False, "reason": "a1 must be free when profiling a varying resonance bank"}

    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    truth_g = np.asarray(g_master[:, obs_idx], dtype=np.float64)
    rms_true = master_rms(g_master)
    basis_m, basis_gamma = _continuous_basis_grid(cfg)
    n_basis = len(basis_m)
    qobs = len(obs_idx)
    physics = make_physics(cfg)
    integration_points = int(cfg["forward"].get("integration_points", 128))
    basis_build_chunk = int(cp.get("basis_build_chunk", 256))

    # P and C are fixed shape vectors. Resonance R changes over (m,gamma).
    backgrounds: dict[str, np.ndarray] = {}
    for name in ("a2", "a3"):
        if name in free_linear or name in cfg["fixed_params"]:
            backgrounds[name] = _unit_linear_component(name, cfg, obs_idx)
    fixed_background = np.zeros(qobs, dtype=np.float64)
    for name in ("a2", "a3"):
        if name in cfg["fixed_params"]:
            fixed_background += float(cfg["fixed_params"][name]) * backgrounds[name]

    response = np.empty((n_basis, qobs), dtype=np.float32)
    a_ref = float(cfg["parameter_ranges"]["a1"][1]) if "a1" in cfg["parameter_ranges"] else 0.2
    for s in range(0, n_basis, basis_build_chunk):
        e = min(s + basis_build_chunk, n_basis)
        p = np.zeros((e - s, 5), dtype=np.float64)
        p[:, 0] = a_ref
        p[:, 3] = basis_m[s:e]
        p[:, 4] = basis_gamma[s:e]
        g = scaled_forward_observation_numpy(p, integration_points=integration_points, config=physics)
        response[s:e] = (np.asarray(g[:, obs_idx], dtype=np.float64) / a_ref).astype(np.float32)
        if s == 0 or e == n_basis or e % max(1, 10 * basis_build_chunk) == 0:
            print(f"[profile-basis] {e} / {n_basis}")

    lo_vec, hi_vec, remote_delta = [], [], []
    ud = cfg["identifiability"]["unacceptable_difference"]
    for name in free_linear:
        lo, hi = [float(v) for v in cfg["parameter_ranges"][name]]
        lo_vec.append(lo); hi_vec.append(hi); remote_delta.append(float(ud[f"{name}_abs"]))
    lo_vec = np.asarray(lo_vec, dtype=np.float64)
    hi_vec = np.asarray(hi_vec, dtype=np.float64)
    remote_delta = np.asarray(remote_delta, dtype=np.float64)

    truth_chunk = int(cp.get("truth_chunk", 16 if len(free_linear) == 3 else 32))
    basis_chunk = int(cp.get("basis_profile_chunk", 1024 if len(free_linear) == 3 else 2048))
    ref_noise = float(cfg["identifiability"]["reference_noise"])
    best_score = np.full(len(params), np.inf, dtype=np.float64)
    best_basis = np.full(len(params), -1, dtype=np.int64)
    best_x = np.full((len(params), len(free_linear)), np.nan, dtype=np.float64)

    for ts in range(0, len(params), truth_chunk):
        te = min(ts + truth_chunk, len(params))
        y = truth_g[ts:te] - fixed_background[None, :]
        yy = np.mean(y * y, axis=1)
        tx = params[ts:te][:, [PARAM_INDEX[n] for n in free_linear]]
        local_best = np.full(te - ts, np.inf, dtype=np.float64)
        local_basis = np.full(te - ts, -1, dtype=np.int64)
        local_x = np.full((te - ts, len(free_linear)), np.nan, dtype=np.float64)
        tm = params[ts:te, PARAM_INDEX["m"]]
        tg = params[ts:te, PARAM_INDEX["gamma"]]

        for bs in range(0, n_basis, basis_chunk):
            be = min(bs + basis_chunk, n_basis)
            rb = np.asarray(response[bs:be], dtype=np.float64)
            comps = []
            for name in free_linear:
                if name == "a1":
                    comps.append(rb)
                else:
                    comps.append(np.broadcast_to(backgrounds[name][None, :], rb.shape))
            X = np.stack(comps, axis=1)  # [B,D,Q]
            G = np.einsum("bdq,beq->bde", X, X, optimize=True) / float(qobs)
            b = np.einsum("tq,bdq->tbd", y, X, optimize=True) / float(qobs)

            nonlinear_remote = np.zeros((te - ts, be - bs), dtype=bool)
            if "m" in cfg["free_params"]:
                nonlinear_remote |= np.abs(tm[:, None] - basis_m[None, bs:be]) >= float(ud["m_abs"])
            if "gamma" in cfg["free_params"]:
                gb = np.maximum(basis_gamma[None, bs:be], 1e-300)
                gt = np.maximum(tg[:, None], 1e-300)
                factor = np.maximum(gb / gt, gt / gb)
                nonlinear_remote |= factor >= float(ud["gamma_factor"])

            xopt, obj = _remote_profile_qp(G, b, tx, lo_vec, hi_vec, remote_delta, nonlinear_remote)
            residual2 = np.maximum(yy[:, None] + obj, 0.0)
            score = np.sqrt(residual2) / np.maximum(ref_noise * rms_true[ts:te, None], 1e-30)
            k = np.argmin(score, axis=1)
            v = score[np.arange(te - ts), k]
            take = v < local_best
            if np.any(take):
                local_best[take] = v[take]
                local_basis[take] = bs + k[take]
                local_x[take] = xopt[np.arange(te - ts), k][take]
        best_score[ts:te] = local_best
        best_basis[ts:te] = local_basis
        best_x[ts:te] = local_x
        print(f"[continuous-profile] {te} / {len(params)}")

    rows: list[dict[str, Any]] = []
    alt_params = np.empty_like(params)
    for i in range(len(params)):
        alt = np.asarray(params[i], dtype=np.float64).copy()
        for name, value in cfg["fixed_params"].items():
            alt[PARAM_INDEX[name]] = float(value)
        for j, name in enumerate(free_linear):
            alt[PARAM_INDEX[name]] = best_x[i, j]
        bi = best_basis[i]
        if "m" in cfg["free_params"]:
            alt[PARAM_INDEX["m"]] = basis_m[bi]
        if "gamma" in cfg["free_params"]:
            alt[PARAM_INDEX["gamma"]] = basis_gamma[bi]
        alt_params[i] = alt
    physical = valid_parameter_mask_numpy(alt_params)
    for i in range(len(params)):
        pm = parameter_pair_metrics(params[i], alt_params[i], cfg)
        row: dict[str, Any] = {
            "state_id": str(state_id[i]),
            "profiled_alias_rms_snr": float(best_score[i]),
            "profiled_g_rms_distance": float(best_score[i] * ref_noise * rms_true[i]),
            "profiled_alias_physical": bool(physical[i]),
            "profiled_alias_is_conservative_lower_bound": bool(not physical[i]),
            "profiled_alias_basis_index": int(best_basis[i]),
        }
        for name in PARAMETER_NAMES:
            row[f"profiled_alias_{name}"] = float(alt_params[i, PARAM_INDEX[name]])
        row.update({f"profiled_{k}": v for k, v in pm.items()})
        rows.append(row)
    return rows, {
        "enabled": True,
        "free_linear": free_linear,
        "basis_count": int(n_basis),
        "m_points": int(len(np.unique(basis_m))),
        "gamma_points": int(len(np.unique(basis_gamma))),
        "truth_chunk": truth_chunk,
        "basis_profile_chunk": basis_chunk,
        "physical_best_alias_fraction": float(np.mean(physical)),
        "score_interpretation": "continuous profiled remote-alias lower bound; nonphysical winning aliases are conservative for screening",
    }
