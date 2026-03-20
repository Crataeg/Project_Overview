from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr.reshape((-1,))


def summarize_distribution(values: Sequence[float], percentiles: Sequence[float] = (10, 50, 90)) -> Dict[str, float]:
    arr = _as_float_array(values)
    if arr.size == 0:
        out: Dict[str, float] = {"count": 0}
        out["mean"] = 0.0
        for p in percentiles:
            out[f"p{int(p)}"] = 0.0
        return out

    out = {"count": int(arr.size), "mean": float(np.mean(arr))}
    for p in percentiles:
        out[f"p{int(p)}"] = float(np.percentile(arr, float(p)))
    out["min"] = float(np.min(arr))
    out["max"] = float(np.max(arr))
    return out


def outage_probability(sinr_db: Sequence[float], thresholds_db: Sequence[float]) -> Dict[str, float]:
    sinr = _as_float_array(sinr_db)
    out: Dict[str, float] = {}
    for th in thresholds_db:
        out[str(float(th))] = float(np.mean(sinr < float(th))) if sinr.size else 0.0
    return out


def tail_mean(values: Sequence[float], tail_pct: float) -> float:
    arr = _as_float_array(values)
    if arr.size == 0:
        return 0.0
    tail_pct = float(np.clip(float(tail_pct), 0.0, 100.0))
    if tail_pct <= 0.0:
        return float(np.mean(arr))
    k = int(max(1, np.floor(arr.size * (tail_pct / 100.0))))
    arr_sorted = np.sort(arr)
    return float(np.mean(arr_sorted[:k]))


@dataclass(frozen=True)
class KpiConfig:
    outage_thresholds_db: Tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0)
    percentiles: Tuple[float, ...] = (10.0, 50.0, 90.0)
    tail_pct: float = 5.0


def compute_kpis(
    link_records: Sequence[Mapping[str, float]],
    cfg: KpiConfig = KpiConfig(),
) -> Dict:
    sinr_db = [float(r.get("sinr_db", 0.0)) for r in link_records]
    rate_bps = [float(r.get("rate_bps", 0.0)) for r in link_records]
    ee_bpj = [float(r.get("ee_bpj", 0.0)) for r in link_records]

    out = {
        "count_links": int(len(link_records)),
        "outage": outage_probability(sinr_db, cfg.outage_thresholds_db),
        "sinr_db": summarize_distribution(sinr_db, cfg.percentiles),
        "rate_bps": summarize_distribution(rate_bps, cfg.percentiles),
        "ee_bpj": summarize_distribution(ee_bpj, cfg.percentiles),
        "tail": {
            "tail_pct": float(cfg.tail_pct),
            "tail_mean_sinr_db": tail_mean(sinr_db, cfg.tail_pct),
            "tail_mean_rate_bps": tail_mean(rate_bps, cfg.tail_pct),
            "tail_mean_ee_bpj": tail_mean(ee_bpj, cfg.tail_pct),
        },
    }
    return out


def bler_interpolate(
    sinr_db: Sequence[float],
    lut_sinr_db: Sequence[float],
    lut_bler: Sequence[float],
) -> np.ndarray:
    x = _as_float_array(sinr_db)
    xp = _as_float_array(lut_sinr_db)
    fp = _as_float_array(lut_bler)
    if x.size == 0:
        return np.zeros((0,), dtype=float)
    if xp.size == 0 or fp.size == 0 or xp.size != fp.size:
        raise ValueError("Invalid BLER LUT: sinr_db and bler must have same non-zero length.")
    order = np.argsort(xp)
    xp = xp[order]
    fp = np.clip(fp[order], 0.0, 1.0)
    y = np.interp(x, xp, fp, left=float(fp[0]), right=float(fp[-1]))
    return np.clip(y, 0.0, 1.0)


def compute_throughput_kpis(
    link_records: Sequence[Mapping[str, float]],
    lut_sinr_db: Sequence[float],
    lut_bler: Sequence[float],
    cfg: KpiConfig = KpiConfig(),
) -> Dict:
    sinr_db = [float(r.get("sinr_db", 0.0)) for r in link_records]
    rate_bps = _as_float_array([float(r.get("rate_bps", 0.0)) for r in link_records])
    if rate_bps.size == 0:
        thr = np.zeros((0,), dtype=float)
        bler = np.zeros((0,), dtype=float)
    else:
        bler = bler_interpolate(sinr_db, lut_sinr_db, lut_bler)
        thr = rate_bps * (1.0 - bler)

    return {
        "count_links": int(len(link_records)),
        "throughput_bps": summarize_distribution(thr.tolist(), cfg.percentiles),
        "tail": {
            "tail_pct": float(cfg.tail_pct),
            "tail_mean_throughput_bps": tail_mean(thr.tolist(), cfg.tail_pct),
        },
    }

