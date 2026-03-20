from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from UAV_GA import DroneCommProblem
from bler_mc import McConfig, save_bler_csv as save_bler_csv_a, simulate_bler_curve
from compare_measured_cornercases import (
    DEFAULT_AERPAW_ZIP,
    DEFAULT_DRYAD_ZIP,
    run_comparison as run_measured_corner_comparison,
)
from kpi import KpiConfig, compute_kpis, compute_throughput_kpis
from plot_results import plot_kpi_comparison


def default_user_params() -> Dict:
    return {
        "num_drones": 6,
        "num_stations": 4,
        "area_size": 1200,
        "drone_height_max": 350,
        "building_height_max": 120,
        "building_density": 0.45,
        "itu_alpha": 0.45,
        "itu_beta": 120.0,
        "itu_gamma": 40.0,
        "drone_speed_max": 18,
        "num_controlled_interference": 8,
    }


def load_user_params(path: str) -> Dict:
    params = default_user_params()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def find_latest_run_id(output_dir: str = "output") -> str:
    if not os.path.isdir(output_dir):
        return ""
    candidates = []
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name)
        if not os.path.isdir(p):
            continue
        if not name.startswith("run_"):
            continue
        candidates.append((os.path.getmtime(p), name))
    candidates.sort()
    return candidates[-1][1] if candidates else ""


def load_samples(path: str) -> np.ndarray:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Samples file not found: {path}")
    return np.load(path)


def discrete_indices(problem: DroneCommProblem) -> Tuple[np.ndarray, np.ndarray]:
    d = int(problem.num_drones)
    m = int(problem.num_interference)
    type_idx = np.arange(4 * d, 4 * d + m)
    loc_idx = np.arange(4 * d + 2 * m, 4 * d + 3 * m)
    return type_idx, loc_idx


def postprocess_x(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, type_idx: np.ndarray, loc_idx: np.ndarray) -> np.ndarray:
    out = np.clip(x, lb, ub)
    if type_idx.size > 0:
        out[:, type_idx] = np.round(out[:, type_idx])
        out[:, type_idx] = np.clip(out[:, type_idx], 0, 7)
    if loc_idx.size > 0:
        out[:, loc_idx] = np.round(out[:, loc_idx])
        out[:, loc_idx] = np.clip(out[:, loc_idx], 0, 1)
    return out


def random_samples(problem: DroneCommProblem, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    x = rng.uniform(lb, ub, size=(int(n), lb.size))
    type_idx, loc_idx = discrete_indices(problem)
    return postprocess_x(x, lb, ub, type_idx, loc_idx)


def _dbm_to_w(power_dbm: float) -> float:
    return float(10.0 ** ((float(power_dbm) - 30.0) / 10.0))


def extract_link_records(
    problem: DroneCommProblem,
    scenario: Mapping,
    scenario_idx: int,
    group: str,
) -> List[Dict[str, float]]:
    drones = np.asarray(scenario.get("drone_positions", np.zeros((0, 3))), dtype=float)
    stations = np.asarray(scenario.get("station_positions", np.zeros((0, 3))), dtype=float)
    speeds = np.asarray(scenario.get("drone_speeds", np.zeros((len(drones),))), dtype=float)
    b_hz = float(getattr(problem, "bandwidth_hz", 0.0))
    tx_dbm = float(getattr(problem, "_comm_tx_power_dbm", 0.0))
    p_tx_w = _dbm_to_w(tx_dbm)

    recs: List[Dict[str, float]] = []
    if drones.size == 0 or stations.size == 0:
        return recs

    for di in range(int(len(drones))):
        v = float(speeds[di]) if di < len(speeds) else 0.0
        p_prop_w = float(problem._rotor_propulsion_power_w(v))
        for si in range(int(len(stations))):
            pm = problem.calculate_power_margin_components(
                drones[di],
                stations[si],
                dict(scenario),
                drone_idx=int(di),
            )
            sinr_lin = float(pm.get("SINR_linear", 0.0))
            sinr_db = float(10.0 * np.log10(max(sinr_lin, 1e-15)))
            rate_bps = float(pm.get("R_bps", 0.0))
            if rate_bps <= 0.0 and b_hz > 0.0:
                rate_bps = float(b_hz * np.log2(1.0 + max(sinr_lin, 0.0)))
            p_total_w = float(max(p_prop_w + p_tx_w, 1e-12))
            ee_bpj = float(rate_bps / p_total_w)
            recs.append(
                {
                    "group": group,
                    "scenario_idx": float(scenario_idx),
                    "drone_idx": float(di),
                    "station_idx": float(si),
                    "bandwidth_hz": float(b_hz),
                    "sinr_db": float(sinr_db),
                    "rate_bps": float(rate_bps),
                    "p_prop_w": float(p_prop_w),
                    "p_tx_w": float(p_tx_w),
                    "ee_bpj": float(ee_bpj),
                }
            )
    return recs


def write_link_csv(path: str, records: Sequence[Mapping[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group", "scenario_idx", "drone_idx", "station_idx", "bandwidth_hz", "sinr_db", "rate_bps", "p_prop_w", "p_tx_w", "ee_bpj"])
        return
    cols = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, 0.0) for k in cols})


def load_bler_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["sinr_db"]))
            ys.append(float(row["bler"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


@dataclass(frozen=True)
class EvaluateConfig:
    outage_thresholds_db: Tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0)
    percentiles: Tuple[float, ...] = (10.0, 50.0, 90.0)
    tail_pct: float = 5.0

    bler_sinr_grid_db: Tuple[float, ...] = (-5.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
    bler_a_mod: str = "qpsk"
    bler_a_bits: int = 1024
    bler_a_blocks: int = 400
    bler_seed: int = 0


def _default_sionna_python() -> str:
    """
    Prefer a dedicated venv for TF+Sionna because the main project venv
    may use an older Python version.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, ".venv_sionna", "Scripts", "python.exe"),
        os.path.join(base_dir, ".venv-tf", "Scripts", "python.exe"),
        os.path.join(base_dir, ".venv_tf", "Scripts", "python.exe"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def _generate_bler_b_via_subprocess(
    sionna_python: str,
    out_csv: str,
    cfg: EvaluateConfig,
) -> None:
    if not sionna_python or not os.path.isfile(sionna_python):
        raise FileNotFoundError(f"Sionna python not found: {sionna_python}")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bler_sionna.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"bler_sionna.py not found: {script}")

    sinr_db_arg = ",".join(str(float(x)) for x in cfg.bler_sinr_grid_db)
    cmd = [
        sionna_python,
        script,
        "--out",
        out_csv,
        f"--sinr_db={sinr_db_arg}",
        "--mod",
        str(cfg.bler_a_mod),
        "--bits",
        str(int(cfg.bler_a_bits)),
        "--blocks",
        str(int(cfg.bler_a_blocks)),
        "--seed",
        str(int(cfg.bler_seed)),
        "--channel",
        "awgn",
    ]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    subprocess.run(cmd, check=True)


def evaluate_groups_from_samples(
    problem: DroneCommProblem,
    groups: Sequence[Tuple[str, np.ndarray]],
    out_dir: str,
    cfg: EvaluateConfig = EvaluateConfig(),
    gen_bler_a: bool = True,
    gen_bler_b: bool = False,
    bler_b_csv: str = "",
    sionna_python: str = "",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    kcfg = KpiConfig(
        outage_thresholds_db=cfg.outage_thresholds_db,
        percentiles=cfg.percentiles,
        tail_pct=float(cfg.tail_pct),
    )

    # BLER A (MC)
    bler_a_path = os.path.join(out_dir, "BLER_A.csv")
    if gen_bler_a or not os.path.isfile(bler_a_path):
        sinr_db, bler = simulate_bler_curve(
            cfg.bler_sinr_grid_db,
            cfg=McConfig(modulation=cfg.bler_a_mod, n_bits_per_block=cfg.bler_a_bits, n_blocks=cfg.bler_a_blocks, seed=cfg.bler_seed),
        )
        save_bler_csv_a(bler_a_path, sinr_db, bler)
    bler_a_x, bler_a_y = load_bler_csv(bler_a_path)

    bler_b_x: Optional[np.ndarray] = None
    bler_b_y: Optional[np.ndarray] = None
    bler_b_path = bler_b_csv
    if gen_bler_b:
        try:
            bler_b_path = os.path.join(out_dir, "BLER_B.csv")
            # Prefer in-process generation if Sionna is importable in current env.
            # Otherwise fall back to a dedicated TF+Sionna venv via subprocess.
            try:
                from bler_sionna import SionnaConfig, save_bler_csv as save_bler_csv_b, simulate_bler_curve_sionna

                sinr_db_b, bler_b = simulate_bler_curve_sionna(
                    cfg.bler_sinr_grid_db,
                    cfg=SionnaConfig(
                        modulation=cfg.bler_a_mod,
                        n_bits_per_block=cfg.bler_a_bits,
                        n_blocks=cfg.bler_a_blocks,
                        seed=cfg.bler_seed,
                        channel="awgn",
                    ),
                )
                save_bler_csv_b(bler_b_path, sinr_db_b, bler_b)
            except Exception:
                sp = sionna_python or os.environ.get("SIONNA_PYTHON", "") or _default_sionna_python()
                _generate_bler_b_via_subprocess(sp, bler_b_path, cfg)
        except Exception as e:
            print(f"[WARN] BLER_B generation skipped: {e}")

    if bler_b_path and os.path.isfile(bler_b_path):
        bler_b_x, bler_b_y = load_bler_csv(bler_b_path)

    results: Dict[str, Dict] = {}
    all_records: Dict[str, List[Dict[str, float]]] = {}
    for group_name, samples in groups:
        records: List[Dict[str, float]] = []
        for sidx, x in enumerate(samples):
            scenario = problem.generate_scenario(np.asarray(x, dtype=float))
            records.extend(extract_link_records(problem, scenario, int(sidx), group_name))
        all_records[group_name] = records
        write_link_csv(os.path.join(out_dir, f"links_{group_name}.csv"), records)

        base = compute_kpis(records, cfg=kcfg)
        thr_a = compute_throughput_kpis(records, bler_a_x, bler_a_y, cfg=kcfg)
        out = {"kpis": base, "throughput_A": thr_a}
        if bler_b_x is not None and bler_b_y is not None:
            out["throughput_B"] = compute_throughput_kpis(records, bler_b_x, bler_b_y, cfg=kcfg)
        results[group_name] = out

    report = {
        "out_dir": out_dir,
        "bler_a_csv": bler_a_path,
        "bler_b_csv": bler_b_path if (bler_b_path and os.path.isfile(bler_b_path)) else "",
        "groups": results,
    }
    with open(os.path.join(out_dir, "kpi_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_kpi_comparison(report, out_dir=out_dir)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="", help="path to JSON with user_params overrides")
    ap.add_argument("--run_id", default="", help="output run folder id")
    ap.add_argument("--ga_samples", default="")
    ap.add_argument("--gan_samples", default="")
    ap.add_argument("--random_n", type=int, default=50)
    ap.add_argument("--random_seed", type=int, default=2024)
    ap.add_argument("--out_dir", default="", help="output folder (default: output/<run_id>/compare/kpi)")

    ap.add_argument("--gen_bler_a", action="store_true", help="(re)generate BLER_A.csv via MC")
    ap.add_argument("--gen_bler_b", action="store_true", help="(re)generate BLER_B.csv via Sionna (if installed)")
    ap.add_argument("--bler_b_csv", default="", help="path to BLER_B.csv (Sionna LUT), optional")
    ap.add_argument("--sionna_python", default="", help="python.exe with TF+Sionna installed (optional)")
    ap.add_argument("--bler_a_mod", default="qpsk", choices=["qpsk", "16qam"])
    ap.add_argument("--bler_a_bits", type=int, default=1024)
    ap.add_argument("--bler_a_blocks", type=int, default=400)
    ap.add_argument("--bler_seed", type=int, default=0)
    ap.add_argument("--measured_corner_compare", action="store_true", help="run measured trajectory corner-case comparison module")
    ap.add_argument("--aerpaw_zip", default=DEFAULT_AERPAW_ZIP, help="path to aerpaw measurement zip")
    ap.add_argument("--dryad_zip", default=DEFAULT_DRYAD_ZIP, help="path to dryad measurement zip")
    ap.add_argument("--corner_random_points", type=int, default=120)
    ap.add_argument("--corner_worst_top_k", type=int, default=25)
    ap.add_argument("--corner_grid_size", type=int, default=55)
    ap.add_argument("--corner_tx_power_dbm", type=float, default=43.0)
    ap.add_argument("--corner_bandwidth_hz", type=float, default=100e6)
    ap.add_argument("--corner_disable_hotspots", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id.strip() or find_latest_run_id()
    if not run_id:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")

    user_params = load_user_params(args.params)
    problem = DroneCommProblem(user_params)

    if args.ga_samples:
        ga_path = args.ga_samples
    else:
        ga_path = os.path.join("output", run_id, "gan", "arrays", "ga_samples.npy")
        if not os.path.isfile(ga_path):
            ga_path = os.path.join("output", "gan_outputs", "ga_samples.npy")

    if args.gan_samples:
        gan_path = args.gan_samples
    else:
        gan_path = os.path.join("output", run_id, "gan", "arrays", "gan_samples.npy")
        if not os.path.isfile(gan_path):
            gan_path = os.path.join("output", "gan_outputs", "gan_samples.npy")

    ga_samples = load_samples(ga_path)
    gan_samples = load_samples(gan_path)
    rnd_samples = random_samples(problem, args.random_n, args.random_seed)

    out_dir = args.out_dir.strip() or os.path.join("output", run_id, "compare", "kpi")
    cfg = EvaluateConfig(
        bler_a_mod=args.bler_a_mod,
        bler_a_bits=int(args.bler_a_bits),
        bler_a_blocks=int(args.bler_a_blocks),
        bler_seed=int(args.bler_seed),
    )
    evaluate_groups_from_samples(
        problem,
        groups=[("ga", ga_samples), ("gan", gan_samples), ("random", rnd_samples)],
        out_dir=out_dir,
        cfg=cfg,
        gen_bler_a=bool(args.gen_bler_a),
        gen_bler_b=bool(args.gen_bler_b),
        bler_b_csv=str(args.bler_b_csv),
        sionna_python=str(args.sionna_python),
    )
    if bool(args.measured_corner_compare):
        measured_out = os.path.join("output", run_id, "compare", "measured_cornercases")
        run_measured_corner_comparison(
            aerpaw_zip=str(args.aerpaw_zip),
            dryad_zip=str(args.dryad_zip),
            out_dir=measured_out,
            random_points=int(args.corner_random_points),
            worst_top_k=int(args.corner_worst_top_k),
            grid_size=int(args.corner_grid_size),
            tx_power_dbm=float(args.corner_tx_power_dbm),
            bandwidth_hz=float(args.corner_bandwidth_hz),
            use_empirical_hotspots=not bool(args.corner_disable_hotspots),
            seed=int(args.random_seed),
        )
    print(f"Saved KPI outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
