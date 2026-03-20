from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from UAV_GA import DroneCommProblem, InterferenceSource
import gan_uav_pipeline as gan_pipeline
from kpi import outage_probability, summarize_distribution, tail_mean


DEFAULT_AERPAW_ZIP = r"d:\Downloads\aerpaw-dataset-24.zip"
DEFAULT_DRYAD_ZIP = r"d:\Downloads\doi_10_5061_dryad_wh70rxx06__v20250521.zip"
DEFAULT_REF_PDF = r"d:\UAV_Communication_GA\key reference\Aerial_RF_and_Throughput_Measurements_on_a_Non-Standalone_5G_Wireless_Network.pdf"

AERPAW_NR_FILES = (
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight1.csv",
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight2.csv",
)
AERPAW_THR_FILES = ("aerpaw-dataset-24/Logs/pawprints_iperf_throughput_altitude30m_flight1.csv",)
DRYAD_SINR_FILES = (
    "Dryad/yaw45/inputf1_sinr_with_header.csv",
    "Dryad/yaw315/inputf1_sinr_with_header.csv",
)
DRYAD_THR_FILES = (
    "Dryad/yaw45/input_throughput_with_header.csv",
    "Dryad/yaw315/input_throughput_with_header.csv",
)


def _base_user_params() -> Dict[str, float]:
    return {
        "num_drones": 1,
        "num_stations": 1,
        "area_size": 800,
        "drone_height_max": 120,
        "building_height_max": 30,
        "building_density": 0.01,
        "itu_alpha": 0.01,
        "itu_beta": 20.0,
        "itu_gamma": 8.0,
        "drone_speed_max": 18,
        "num_controlled_interference": 8,
        "background_interference_enabled": False,
        "bandwidth_hz": 100e6,
        "noise_figure_db": 7.0,
        "thermal_noise_dbm_hz": -174.0,
        "los_indicator_mode": "deterministic_if_map",
        "spectral_mode": "acir",
        "objective_model": "power_score",
    }


@dataclass(frozen=True)
class MeasuredPoint:
    dataset_id: str
    track_id: str
    selection_id: str
    selection_role: str
    source_zip_path: str
    internal_path: str
    row_number: int
    metric_name: str
    raw_metric_value: float
    longitude: float
    latitude: float
    altitude_m: float
    x_local_m: float
    y_local_m: float
    z_local_m: float
    x_aligned_m: float = 0.0
    y_aligned_m: float = 0.0
    bs_distance_m: float = float("nan")
    bs_bearing_deg: float = float("nan")
    bs_elevation_deg: float = float("nan")
    note: str = ""


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    family: str
    reference_altitude_m: float
    station_height_m: float
    comm_freq_hz: float
    notes: str
    all_points: Tuple[MeasuredPoint, ...]
    selected_points: Tuple[MeasuredPoint, ...]
    station_xy_local_m: Tuple[float, float]


@dataclass(frozen=True)
class EnvironmentBundle:
    dataset_id: str
    area_size_m: float
    station_position_m: Tuple[float, float, float]
    station_xy_local_m: Tuple[float, float]
    station_xy_aligned_m: Tuple[float, float]
    shift_xy_m: Tuple[float, float]
    hotspot_specs: Tuple[Dict[str, float], ...]
    selected_points: Tuple[MeasuredPoint, ...]
    all_points: Tuple[MeasuredPoint, ...]
    notes: str


def _metric_suffix(metric_name: str) -> str:
    return "sinr" if "sinr" in str(metric_name).lower() else "throughput"


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _csv_rows_from_zip(zip_path: str, internal_path: str) -> Iterable[Tuple[int, Mapping[str, str]]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(internal_path) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            for idx, row in enumerate(reader, start=2):
                yield idx, row


def _csv_rows_from_nested_zip(outer_zip_path: str, nested_zip_name: str, internal_path: str) -> Iterable[Tuple[int, Mapping[str, str]]]:
    with zipfile.ZipFile(outer_zip_path) as outer:
        data = io.BytesIO(outer.read(nested_zip_name))
    with zipfile.ZipFile(data) as inner:
        with inner.open(internal_path) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            for idx, row in enumerate(reader, start=2):
                yield idx, row


def _bearing_distance_to_local_xy(distance_m: float, bearing_deg: float, elevation_deg: float) -> Tuple[float, float]:
    horizontal = float(distance_m) * math.cos(math.radians(float(elevation_deg)))
    bearing_rad = math.radians(float(bearing_deg))
    x = horizontal * math.sin(bearing_rad)
    y = horizontal * math.cos(bearing_rad)
    return float(x), float(y)


def _latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    r_earth = 6371000.0
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=float))
    lat0_rad = math.radians(float(lat0))
    lon0_rad = math.radians(float(lon0))
    x = r_earth * (lon_rad - lon0_rad) * math.cos(lat0_rad)
    y = r_earth * (lat_rad - math.radians(float(lat0)))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _spatial_distance(a: MeasuredPoint, b: MeasuredPoint) -> float:
    return float(math.hypot(float(a.x_local_m) - float(b.x_local_m), float(a.y_local_m) - float(b.y_local_m)))


def _pick_unique(
    points: Sequence[MeasuredPoint],
    sort_key,
    reverse: bool,
    min_separation_m: float,
    existing: Optional[Sequence[MeasuredPoint]] = None,
    limit: int = 1,
) -> List[MeasuredPoint]:
    chosen: List[MeasuredPoint] = []
    occupied = list(existing or [])
    ranked = sorted(points, key=sort_key, reverse=reverse)
    for point in ranked:
        ok = True
        for prev in occupied:
            if _spatial_distance(point, prev) < float(min_separation_m):
                ok = False
                break
        if ok:
            chosen.append(point)
            occupied.append(point)
        if len(chosen) >= int(limit):
            break
    return chosen


def _infer_aerpaw_station_height(zip_path: str) -> float:
    heights: List[float] = []
    for internal_path in AERPAW_NR_FILES:
        for _, row in _csv_rows_from_zip(zip_path, internal_path):
            alt = _safe_float(row.get("altitude"))
            dist = _safe_float(row.get("bs_distance"))
            elev = _safe_float(row.get("bs_elevation"))
            if not np.isfinite(alt) or not np.isfinite(dist) or not np.isfinite(elev):
                continue
            horizontal = dist * math.cos(math.radians(elev))
            station_height = alt - horizontal * math.tan(math.radians(elev))
            if np.isfinite(station_height):
                heights.append(float(station_height))
    if not heights:
        return 12.0
    return float(np.median(np.asarray(heights, dtype=float)))


def _load_aerpaw_bundle(zip_path: str) -> DatasetBundle:
    station_height_m = _infer_aerpaw_station_height(zip_path)
    points: List[MeasuredPoint] = []

    for internal_path in AERPAW_NR_FILES:
        track_id = Path(internal_path).stem
        for row_number, row in _csv_rows_from_zip(zip_path, internal_path):
            metric = _safe_float(row.get("ss_sinr"))
            dist = _safe_float(row.get("bs_distance"))
            bearing = _safe_float(row.get("bs_bearing"))
            elev = _safe_float(row.get("bs_elevation"))
            if not np.isfinite(metric) or not np.isfinite(dist) or not np.isfinite(bearing):
                continue
            x_m, y_m = _bearing_distance_to_local_xy(dist, bearing, elev if np.isfinite(elev) else 0.0)
            points.append(
                MeasuredPoint(
                    dataset_id="aerpaw",
                    track_id=track_id,
                    selection_id="",
                    selection_role="candidate",
                    source_zip_path=zip_path,
                    internal_path=internal_path,
                    row_number=row_number,
                    metric_name="measured_sinr_db",
                    raw_metric_value=float(metric),
                    longitude=_safe_float(row.get("longitude")),
                    latitude=_safe_float(row.get("latitude")),
                    altitude_m=_safe_float(row.get("altitude")),
                    x_local_m=float(x_m),
                    y_local_m=float(y_m),
                    z_local_m=_safe_float(row.get("altitude")),
                    bs_distance_m=dist,
                    bs_bearing_deg=bearing,
                    bs_elevation_deg=elev,
                )
            )

    for internal_path in AERPAW_THR_FILES:
        track_id = Path(internal_path).stem
        for row_number, row in _csv_rows_from_zip(zip_path, internal_path):
            metric = _safe_float(row.get("iperf_client_mbps"))
            dist = _safe_float(row.get("bs_distance"))
            bearing = _safe_float(row.get("bs_bearing"))
            elev = _safe_float(row.get("bs_elevation"))
            if not np.isfinite(metric) or not np.isfinite(dist) or not np.isfinite(bearing):
                continue
            x_m, y_m = _bearing_distance_to_local_xy(dist, bearing, elev if np.isfinite(elev) else 0.0)
            points.append(
                MeasuredPoint(
                    dataset_id="aerpaw",
                    track_id=track_id,
                    selection_id="",
                    selection_role="candidate",
                    source_zip_path=zip_path,
                    internal_path=internal_path,
                    row_number=row_number,
                    metric_name="measured_throughput_mbps",
                    raw_metric_value=float(metric),
                    longitude=_safe_float(row.get("longitude")),
                    latitude=_safe_float(row.get("latitude")),
                    altitude_m=_safe_float(row.get("altitude")),
                    x_local_m=float(x_m),
                    y_local_m=float(y_m),
                    z_local_m=_safe_float(row.get("altitude")),
                    bs_distance_m=dist,
                    bs_bearing_deg=bearing,
                    bs_elevation_deg=elev,
                )
            )

    selected: List[MeasuredPoint] = []
    poor_f1 = [p for p in points if p.metric_name == "measured_sinr_db" and "flight1" in p.track_id]
    poor_f2 = [p for p in points if p.metric_name == "measured_sinr_db" and "flight2" in p.track_id]
    poor_thr = [p for p in points if p.metric_name == "measured_throughput_mbps"]

    pick = _pick_unique(poor_f1, lambda item: item.raw_metric_value, False, 40.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P1", selection_role="poor_sinr", note="flight1最低SINR点"))

    pick = _pick_unique(poor_f2, lambda item: item.raw_metric_value, False, 40.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P2", selection_role="poor_sinr", note="flight2最低SINR点"))

    pick = _pick_unique(poor_thr, lambda item: item.raw_metric_value, False, 35.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P3", selection_role="poor_throughput", note="吞吐最低点"))

    pick = _pick_unique(points, lambda item: _safe_float(item.bs_distance_m, -1.0), True, 35.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_C1", selection_role="classic_edge", note="最远小区边缘点"))

    pick = _pick_unique(
        [p for p in points if np.isfinite(p.bs_elevation_deg)],
        lambda item: item.bs_elevation_deg,
        False,
        35.0,
        selected,
        1,
    )
    if pick:
        selected.append(replace(pick[0], selection_id="AE_C2", selection_role="classic_low_elevation", note="最小仰角点"))

    altitude_values = [p.altitude_m for p in points if np.isfinite(p.altitude_m)]
    reference_altitude_m = float(np.median(np.asarray(altitude_values, dtype=float))) if altitude_values else 30.0
    return DatasetBundle(
        dataset_id="aerpaw",
        family="AERPAW 5G NSA",
        reference_altitude_m=reference_altitude_m,
        station_height_m=station_height_m,
        comm_freq_hz=3.6e9,
        notes="按日志中的基站距离/方位/仰角重建BS相对坐标；站高由日志反推中位数约12 m。",
        all_points=tuple(points),
        selected_points=tuple(selected),
        station_xy_local_m=(0.0, 0.0),
    )


def _load_dryad_bundle(zip_path: str) -> DatasetBundle:
    records: List[Dict[str, object]] = []
    for internal_path in DRYAD_SINR_FILES:
        track_id = Path(internal_path).parent.name + "_" + Path(internal_path).stem
        for row_number, row in _csv_rows_from_nested_zip(zip_path, "Ericsson_Amir.zip", internal_path):
            metric = _safe_float(row.get("LTE SINR"))
            if not np.isfinite(metric):
                continue
            records.append(
                {
                    "track_id": track_id,
                    "internal_path": internal_path,
                    "row_number": row_number,
                    "metric_name": "measured_sinr_db",
                    "raw_metric_value": float(metric),
                    "longitude": _safe_float(row.get("Longitude")),
                    "latitude": _safe_float(row.get("Latitude")),
                    "altitude_m": _safe_float(row.get("Altitude")),
                }
            )
    for internal_path in DRYAD_THR_FILES:
        track_id = Path(internal_path).parent.name + "_" + Path(internal_path).stem
        for row_number, row in _csv_rows_from_nested_zip(zip_path, "Ericsson_Amir.zip", internal_path):
            metric = _safe_float(row.get("Throughput"))
            if not np.isfinite(metric):
                continue
            records.append(
                {
                    "track_id": track_id,
                    "internal_path": internal_path,
                    "row_number": row_number,
                    "metric_name": "measured_throughput_mbps",
                    "raw_metric_value": float(metric),
                    "longitude": _safe_float(row.get("Longitude")),
                    "latitude": _safe_float(row.get("Latitude")),
                    "altitude_m": _safe_float(row.get("Altitude")),
                }
            )

    lat = np.asarray([float(item["latitude"]) for item in records], dtype=float)
    lon = np.asarray([float(item["longitude"]) for item in records], dtype=float)
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    x_m, y_m = _latlon_to_xy_m(lat, lon, lat0, lon0)

    points: List[MeasuredPoint] = []
    for idx, item in enumerate(records):
        points.append(
            MeasuredPoint(
                dataset_id="dryad",
                track_id=str(item["track_id"]),
                selection_id="",
                selection_role="candidate",
                source_zip_path=zip_path,
                internal_path=str(item["internal_path"]),
                row_number=int(item["row_number"]),
                metric_name=str(item["metric_name"]),
                raw_metric_value=float(item["raw_metric_value"]),
                longitude=float(item["longitude"]),
                latitude=float(item["latitude"]),
                altitude_m=float(item["altitude_m"]),
                x_local_m=float(x_m[idx]),
                y_local_m=float(y_m[idx]),
                z_local_m=float(item["altitude_m"]),
            )
        )

    sinr_points = [p for p in points if p.metric_name == "measured_sinr_db"]
    thr_points = [p for p in points if p.metric_name == "measured_throughput_mbps"]

    good_points: List[MeasuredPoint] = []
    for group in (sinr_points, thr_points):
        if not group:
            continue
        values = np.asarray([p.raw_metric_value for p in group], dtype=float)
        cutoff = float(np.percentile(values, 90.0))
        good_points.extend([p for p in group if p.raw_metric_value >= cutoff])
    if good_points:
        station_x = float(np.mean([p.x_local_m for p in good_points]))
        station_y = float(np.mean([p.y_local_m for p in good_points]))
    else:
        station_x = 0.0
        station_y = 0.0

    selected: List[MeasuredPoint] = []
    poor_sinr_45 = [p for p in sinr_points if "yaw45" in p.track_id]
    poor_sinr_315 = [p for p in sinr_points if "yaw315" in p.track_id]
    poor_thr_45 = [p for p in thr_points if "yaw45" in p.track_id]
    poor_thr_315 = [p for p in thr_points if "yaw315" in p.track_id]

    pick = _pick_unique(poor_sinr_45, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P1", selection_role="poor_sinr", note="yaw45最低SINR点"))

    pick = _pick_unique(poor_sinr_315, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P2", selection_role="poor_sinr", note="yaw315最低SINR点"))

    pick = _pick_unique(poor_thr_45, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P3", selection_role="poor_throughput", note="yaw45最低吞吐点"))

    pick = _pick_unique(poor_thr_315, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P4", selection_role="poor_throughput", note="yaw315最低吞吐点"))

    pick = _pick_unique(points, lambda item: item.x_local_m + item.y_local_m, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_C1", selection_role="classic_corner", note="轨迹西南角点"))

    pick = _pick_unique(points, lambda item: item.x_local_m + item.y_local_m, True, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_C2", selection_role="classic_corner", note="轨迹东北角点"))

    altitude_values = [p.altitude_m for p in points if np.isfinite(p.altitude_m)]
    reference_altitude_m = float(np.median(np.asarray(altitude_values, dtype=float))) if altitude_values else 30.0
    return DatasetBundle(
        dataset_id="dryad",
        family="Dryad Ericsson 5G NSA",
        reference_altitude_m=reference_altitude_m,
        station_height_m=12.0,
        comm_freq_hz=3.6e9,
        notes="Dryad日志无BS方位字段，使用高SINR/高吞吐样本的质心作为服务中心代理；站高沿用AERPAW反推的12 m口径。",
        all_points=tuple(points),
        selected_points=tuple(selected),
        station_xy_local_m=(station_x, station_y),
    )


def _align_bundle(bundle: DatasetBundle, area_margin_m: float, use_empirical_hotspots: bool) -> EnvironmentBundle:
    all_points = list(bundle.all_points)
    selected = list(bundle.selected_points)
    xs = [p.x_local_m for p in all_points] + [float(bundle.station_xy_local_m[0])]
    ys = [p.y_local_m for p in all_points] + [float(bundle.station_xy_local_m[1])]
    min_x = float(np.min(np.asarray(xs, dtype=float)))
    max_x = float(np.max(np.asarray(xs, dtype=float)))
    min_y = float(np.min(np.asarray(ys, dtype=float)))
    max_y = float(np.max(np.asarray(ys, dtype=float)))
    span_x = max_x - min_x
    span_y = max_y - min_y
    span = max(span_x, span_y, 100.0)
    area_size = float(span + 2.0 * area_margin_m)
    shift_x = float(area_margin_m - min_x + 0.5 * max(0.0, span - span_x))
    shift_y = float(area_margin_m - min_y + 0.5 * max(0.0, span - span_y))

    def shift_point(point: MeasuredPoint) -> MeasuredPoint:
        return replace(
            point,
            x_aligned_m=float(point.x_local_m + shift_x),
            y_aligned_m=float(point.y_local_m + shift_y),
        )

    aligned_all = tuple(shift_point(point) for point in all_points)
    aligned_selected = tuple(shift_point(point) for point in selected)
    station_xy_aligned = (
        float(bundle.station_xy_local_m[0] + shift_x),
        float(bundle.station_xy_local_m[1] + shift_y),
    )

    hotspot_specs: List[Dict[str, float]] = []
    if use_empirical_hotspots:
        poor_selected = [p for p in aligned_selected if p.selection_role.startswith("poor")]
        hotspot_points = _pick_unique(
            poor_selected,
            lambda item: item.raw_metric_value,
            False,
            50.0,
            [],
            min(2, len(poor_selected)),
        )
        hotspot_powers = (30.0, 34.0)
        for idx, point in enumerate(hotspot_points):
            hotspot_specs.append(
                {
                    "x_m": float(point.x_aligned_m),
                    "y_m": float(point.y_aligned_m),
                    "height_m": float(bundle.station_height_m + 4.0),
                    "power_dbm": float(hotspot_powers[idx]),
                    "type_id": 3.0,
                }
            )

    return EnvironmentBundle(
        dataset_id=bundle.dataset_id,
        area_size_m=area_size,
        station_position_m=(float(station_xy_aligned[0]), float(station_xy_aligned[1]), float(bundle.station_height_m)),
        station_xy_local_m=bundle.station_xy_local_m,
        station_xy_aligned_m=station_xy_aligned,
        shift_xy_m=(shift_x, shift_y),
        hotspot_specs=tuple(hotspot_specs),
        selected_points=aligned_selected,
        all_points=aligned_all,
        notes=bundle.notes,
    )


def _build_problem(env: EnvironmentBundle, reference_altitude_m: float, comm_freq_hz: float, bandwidth_hz: float, tx_power_dbm: float) -> DroneCommProblem:
    params = _base_user_params()
    params["area_size"] = float(env.area_size_m)
    params["drone_height_max"] = float(max(reference_altitude_m + 30.0, 80.0))
    params["bandwidth_hz"] = float(bandwidth_hz)
    problem = DroneCommProblem(params)
    problem.communication_freq = float(comm_freq_hz)
    problem._comm_tx_power_dbm = float(tx_power_dbm)
    problem.station_positions = np.asarray([env.station_position_m], dtype=float)
    problem.buildings = []
    problem._build_building_spatial_index()
    return problem


def _build_hotspots(env: EnvironmentBundle) -> List[InterferenceSource]:
    hotspots: List[InterferenceSource] = []
    for spec in env.hotspot_specs:
        hotspots.append(
            InterferenceSource(
                float(spec["x_m"]),
                float(spec["y_m"]),
                float(spec["height_m"]),
                int(spec["type_id"]),
                float(spec["power_dbm"]),
                building=None,
                is_ground=False,
                building_idx=None,
                skip_height_clip=True,
            )
        )
    return hotspots


def _build_hotspots_from_specs(hotspot_specs: Sequence[Mapping[str, float]]) -> List[InterferenceSource]:
    hotspots: List[InterferenceSource] = []
    for spec in hotspot_specs:
        hotspots.append(
            InterferenceSource(
                float(spec["x_m"]),
                float(spec["y_m"]),
                float(spec["height_m"]),
                int(spec["type_id"]),
                float(spec["power_dbm"]),
                building=None,
                is_ground=False,
                building_idx=None,
                skip_height_clip=True,
            )
        )
    return hotspots


def _attach_aligned_environment(
    problem: DroneCommProblem,
    env: EnvironmentBundle,
    hotspot_specs: Sequence[Mapping[str, float]],
) -> DroneCommProblem:
    problem.station_positions = np.asarray([env.station_position_m], dtype=float)
    problem.buildings = []
    problem._build_building_spatial_index()

    original_generate_scenario = problem.generate_scenario
    hotspot_specs_tuple = tuple(dict(spec) for spec in hotspot_specs)

    def _aligned_generate_scenario(self: DroneCommProblem, x: np.ndarray) -> Dict:
        scenario = original_generate_scenario(np.asarray(x, dtype=float))
        scenario["station_positions"] = np.asarray([env.station_position_m], dtype=float)
        scenario["buildings"] = []
        fixed_sources = _build_hotspots_from_specs(hotspot_specs_tuple)
        if fixed_sources:
            controlled = list(scenario.get("interference_sources", []))
            scenario["interference_sources"] = controlled + fixed_sources
            viz_sources = list(scenario.get("interference_sources_viz", controlled))
            scenario["interference_sources_viz"] = viz_sources + fixed_sources
            scenario["fixed_alignment_sources"] = fixed_sources
        scenario["interference_power_mw_per_drone"] = self._precompute_interference_power_mw_per_drone(
            drone_positions=scenario["drone_positions"],
            interference_sources=scenario.get("interference_sources", []),
            buildings=scenario.get("buildings", []),
        )
        return scenario

    problem.generate_scenario = MethodType(_aligned_generate_scenario, problem)
    return problem


def _base_scenario(problem: DroneCommProblem, env: EnvironmentBundle, hotspots: Sequence[InterferenceSource]) -> Dict[str, object]:
    return {
        "station_positions": np.asarray([env.station_position_m], dtype=float),
        "interference_sources": list(hotspots),
        "buildings": [],
        "area_size": float(env.area_size_m),
        "building_density": 0.0,
    }


def _evaluate_positions(
    problem: DroneCommProblem,
    scenario: Mapping[str, object],
    positions_xyz: np.ndarray,
) -> List[Dict[str, float]]:
    station_pos = np.asarray(scenario["station_positions"], dtype=float)[0]
    p_prop_w = float(problem._rotor_propulsion_power_w(0.0))
    p_tx_w = float(10.0 ** ((float(problem._comm_tx_power_dbm) - 30.0) / 10.0))
    results: List[Dict[str, float]] = []
    for pos in np.asarray(positions_xyz, dtype=float):
        comp = problem.calculate_power_margin_components(pos, station_pos, dict(scenario), drone_idx=None)
        sinr_db = float(10.0 * np.log10(max(float(comp.get("SINR_linear", 0.0)), 1e-15)))
        rate_bps = float(comp.get("R_bps", 0.0))
        results.append(
            {
                "x_m": float(pos[0]),
                "y_m": float(pos[1]),
                "z_m": float(pos[2]),
                "margin_db": float(comp.get("M_db", 0.0)),
                "sinr_db": float(sinr_db),
                "rate_bps": float(rate_bps),
                "outage": float(comp.get("Outage", 0.0)),
                "i_dbm": float(comp.get("I_y_dbm", -np.inf)),
                "p_rx_dbm": float(comp.get("P_rx_dbm", -np.inf)),
                "ee_bpj": float(rate_bps / max(p_prop_w + p_tx_w, 1e-12)),
            }
        )
    return results


def _search_worst_region(
    problem: DroneCommProblem,
    scenario: Mapping[str, object],
    altitude_m: float,
    area_size_m: float,
    grid_size: int,
    top_k: int,
) -> List[Dict[str, float]]:
    xs = np.linspace(0.0, float(area_size_m), int(grid_size))
    ys = np.linspace(0.0, float(area_size_m), int(grid_size))
    mesh = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    positions = np.column_stack([mesh, np.full((mesh.shape[0],), float(altitude_m), dtype=float)])
    evaluated = _evaluate_positions(problem, scenario, positions)
    evaluated.sort(key=lambda item: (item["sinr_db"], item["rate_bps"]))
    return evaluated[: int(top_k)]


def _sample_random_positions(
    altitude_m: float,
    area_size_m: float,
    n_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    xy = rng.uniform(0.0, float(area_size_m), size=(int(n_points), 2))
    return np.column_stack([xy, np.full((xy.shape[0],), float(altitude_m), dtype=float)])


def _random_candidate_samples(problem: DroneCommProblem, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    samples = rng.uniform(lb, ub, size=(int(n), lb.size))
    type_idx, loc_idx = gan_pipeline.discrete_indices(problem)
    return gan_pipeline.postprocess_x(samples, lb, ub, type_idx, loc_idx)


def _evaluate_candidate_scores(problem: DroneCommProblem, samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0,), dtype=float)
    values, _ = problem.evalVars(np.asarray(samples, dtype=float))
    return np.asarray(values, dtype=float).reshape((-1,))


def _summarize_search_scores(scores: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(scores), dtype=float).reshape((-1,))
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _run_original_model_groups(
    problem: DroneCommProblem,
    seed: int,
    ga_runs: int,
    ga_maxgen: int,
    ga_nind: int,
    gan_epochs: int,
    gan_batch: int,
    gan_latent: int,
    gan_samples_n: int,
    random_search_n: int,
    device: str,
) -> Dict[str, Dict[str, object]]:
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    type_idx, loc_idx = gan_pipeline.discrete_indices(problem)

    ga_samples = gan_pipeline.run_ga_samples(problem, ga_runs, ga_maxgen, ga_nind, seed)
    ga_scores = _evaluate_candidate_scores(problem, ga_samples)
    ga_best_idx = int(np.argmax(ga_scores)) if ga_scores.size else -1

    out: Dict[str, Dict[str, object]] = {
        "ga": {
            "samples": ga_samples,
            "scores": ga_scores,
            "best_x": np.asarray(ga_samples[ga_best_idx], dtype=float) if ga_best_idx >= 0 else np.zeros((problem.Dim,), dtype=float),
            "best_score": float(ga_scores[ga_best_idx]) if ga_best_idx >= 0 else 0.0,
            "score_stats": _summarize_search_scores(ga_scores.tolist()),
        }
    }

    rnd_samples = _random_candidate_samples(problem, random_search_n, seed + 7919)
    rnd_scores = _evaluate_candidate_scores(problem, rnd_samples)
    rnd_best_idx = int(np.argmax(rnd_scores)) if rnd_scores.size else -1
    out["random"] = {
        "samples": rnd_samples,
        "scores": rnd_scores,
        "best_x": np.asarray(rnd_samples[rnd_best_idx], dtype=float) if rnd_best_idx >= 0 else np.zeros((problem.Dim,), dtype=float),
        "best_score": float(rnd_scores[rnd_best_idx]) if rnd_best_idx >= 0 else 0.0,
        "score_stats": _summarize_search_scores(rnd_scores.tolist()),
    }

    if gan_pipeline.torch is not None and ga_samples.size > 0:
        gen, gan_log = gan_pipeline.train_gan(
            ga_samples,
            lb,
            ub,
            int(gan_epochs),
            int(gan_batch),
            int(gan_latent),
            str(device),
        )
        gan_samples = gan_pipeline.generate_gan_samples(
            gen,
            int(gan_samples_n),
            int(gan_latent),
            lb,
            ub,
            type_idx,
            loc_idx,
            str(device),
        )
        gan_scores = _evaluate_candidate_scores(problem, gan_samples)
        gan_best_idx = int(np.argmax(gan_scores)) if gan_scores.size else -1
        out["gan"] = {
            "samples": gan_samples,
            "scores": gan_scores,
            "best_x": np.asarray(gan_samples[gan_best_idx], dtype=float) if gan_best_idx >= 0 else np.zeros((problem.Dim,), dtype=float),
            "best_score": float(gan_scores[gan_best_idx]) if gan_best_idx >= 0 else 0.0,
            "score_stats": _summarize_search_scores(gan_scores.tolist()),
            "gan_training": gan_log,
        }
    return out


def _summarize_model_group(records: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    sinr_db = [float(item["sinr_db"]) for item in records]
    rate_mbps = [float(item["rate_bps"]) / 1e6 for item in records]
    margin_db = [float(item["margin_db"]) for item in records]
    return {
        "count": int(len(records)),
        "sinr_db": summarize_distribution(sinr_db, percentiles=(10, 50, 90)),
        "margin_db": summarize_distribution(margin_db, percentiles=(10, 50, 90)),
        "rate_mbps": summarize_distribution(rate_mbps, percentiles=(10, 50, 90)),
        "outage": outage_probability(sinr_db, thresholds_db=(-5.0, 0.0, 5.0, 10.0)),
        "tail": {
            "tail_mean_sinr_db": float(tail_mean(sinr_db, 5.0)),
            "tail_mean_rate_mbps": float(tail_mean(rate_mbps, 5.0)),
            "tail_mean_margin_db": float(tail_mean(margin_db, 5.0)),
        },
    }


def _extract_dense_reference_points(points: Sequence[MeasuredPoint], quantile: float = 0.1) -> List[MeasuredPoint]:
    dense: List[MeasuredPoint] = []
    for suffix in ("sinr", "throughput"):
        subset = [p for p in points if _metric_suffix(p.metric_name) == suffix]
        if not subset:
            continue
        values = np.asarray([p.raw_metric_value for p in subset], dtype=float)
        threshold = float(np.quantile(values, float(quantile)))
        dense.extend([p for p in subset if p.raw_metric_value <= threshold])
    return dense


def _evaluate_grid_map(
    problem: DroneCommProblem,
    scenario: Mapping[str, object],
    altitude_m: float,
    area_size_m: float,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    xs = np.linspace(0.0, float(area_size_m), int(grid_size))
    ys = np.linspace(0.0, float(area_size_m), int(grid_size))
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    positions = np.column_stack([
        mesh_x.reshape((-1,)),
        mesh_y.reshape((-1,)),
        np.full((mesh_x.size,), float(altitude_m), dtype=float),
    ])
    records = _evaluate_positions(problem, scenario, positions)
    sinr_map = np.asarray([float(item["sinr_db"]) for item in records], dtype=float).reshape(mesh_x.shape)
    return xs, ys, sinr_map, records


def _plot_spatial_layout(
    env: EnvironmentBundle,
    scenario: Mapping[str, object],
    group_name: str,
    random_records: Sequence[Mapping[str, float]],
    worst_records: Sequence[Mapping[str, float]],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    all_x = np.asarray([p.x_aligned_m for p in env.all_points], dtype=float)
    all_y = np.asarray([p.y_aligned_m for p in env.all_points], dtype=float)
    ax.scatter(all_x, all_y, s=8, alpha=0.10, color="#5B6C8F", label="Measured trajectory points")

    poor = [p for p in env.selected_points if p.selection_role.startswith("poor")]
    classic = [p for p in env.selected_points if not p.selection_role.startswith("poor")]
    if poor:
        ax.scatter(
            [p.x_aligned_m for p in poor],
            [p.y_aligned_m for p in poor],
            s=70,
            c="#D62728",
            marker="o",
            label="Measured poor points",
        )
    if classic:
        ax.scatter(
            [p.x_aligned_m for p in classic],
            [p.y_aligned_m for p in classic],
            s=90,
            c="#1F77B4",
            marker="s",
            label="Measured classic points",
        )
    for point in env.selected_points:
        ax.annotate(point.selection_id, (point.x_aligned_m, point.y_aligned_m), textcoords="offset points", xytext=(5, 5), fontsize=8)

    if random_records:
        ax.scatter(
            [float(item["x_m"]) for item in random_records],
            [float(item["y_m"]) for item in random_records],
            s=20,
            alpha=0.35,
            color="#2CA02C",
            label="Random points",
        )
    if worst_records:
        ax.scatter(
            [float(item["x_m"]) for item in worst_records],
            [float(item["y_m"]) for item in worst_records],
            s=70,
            marker="x",
            color="#9467BD",
            label="Model worst region",
        )

    sources = list(scenario.get("interference_sources", []))
    if sources:
        ax.scatter(
            [float(getattr(src, "x", 0.0)) for src in sources],
            [float(getattr(src, "y", 0.0)) for src in sources],
            s=34,
            marker="^",
            color="#FF7F0E",
            alpha=0.75,
            label="Scenario interferers",
        )

    if scenario.get("drone_positions") is not None:
        drones = np.asarray(scenario.get("drone_positions", np.zeros((0, 3))), dtype=float)
        if drones.size:
            ax.scatter(
                drones[:, 0],
                drones[:, 1],
                s=110,
                marker="D",
                color="#8A2BE2",
                edgecolors="black",
                linewidths=0.8,
                label="Optimized UAV position",
            )

    if env.hotspot_specs:
        ax.scatter(
            [float(item["x_m"]) for item in env.hotspot_specs],
            [float(item["y_m"]) for item in env.hotspot_specs],
            s=110,
            marker="^",
            color="#FF7F0E",
            label="Empirical hotspots",
        )

    ax.scatter([env.station_xy_aligned_m[0]], [env.station_xy_aligned_m[1]], s=180, marker="*", color="black", label="Serving BS proxy")
    ax.set_xlim(0.0, float(env.area_size_m))
    ax.set_ylim(0.0, float(env.area_size_m))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("{0}-{1}: measured points vs optimized scene".format(env.dataset_id.upper(), group_name.upper()))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_metric_comparison(
    dataset_id: str,
    group_name: str,
    measured_records: Sequence[Mapping[str, float]],
    random_records: Sequence[Mapping[str, float]],
    worst_records: Sequence[Mapping[str, float]],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    groups = {
        "MeasuredPts": measured_records,
        "Random": random_records,
        "ModelWorst": worst_records,
    }
    labels = list(groups.keys())
    colors = ["#D62728", "#2CA02C", "#9467BD"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    sinr_data = [[float(item["sinr_db"]) for item in groups[label]] for label in labels]
    margin_data = [[float(item["margin_db"]) for item in groups[label]] for label in labels]
    rate_data = [[float(item["rate_bps"]) / 1e6 for item in groups[label]] for label in labels]

    axes[0, 0].boxplot(sinr_data, labels=labels, showfliers=False)
    axes[0, 0].set_title("Model SINR (dB)")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].boxplot(margin_data, labels=labels, showfliers=False)
    axes[0, 1].set_title("Model Margin (dB)")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].boxplot(rate_data, labels=labels, showfliers=False)
    axes[1, 0].set_title("Model Rate (Mbps)")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    thresholds = np.asarray([-5.0, 0.0, 5.0, 10.0], dtype=float)
    for idx, label in enumerate(labels):
        outage = outage_probability([float(item["sinr_db"]) for item in groups[label]], thresholds.tolist())
        axes[1, 1].plot(thresholds, [float(outage[str(float(x))]) for x in thresholds], marker="o", color=colors[idx], label=label)
    axes[1, 1].set_title("Outage curve")
    axes[1, 1].set_xlabel("Threshold (dB)")
    axes[1, 1].set_ylabel("P(SINR < threshold)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("{0}-{1}: model-side comparison".format(dataset_id.upper(), group_name.upper()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_dense_heatmap(
    env: EnvironmentBundle,
    scenario: Mapping[str, object],
    group_name: str,
    dense_reference_points: Sequence[MeasuredPoint],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sinr_map: np.ndarray,
    worst_records: Sequence[Mapping[str, float]],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    extent = [float(grid_x[0]), float(grid_x[-1]), float(grid_y[0]), float(grid_y[-1])]
    image = ax.imshow(
        sinr_map,
        origin="lower",
        extent=extent,
        cmap="turbo",
        aspect="auto",
    )
    plt.colorbar(image, ax=ax, label="Model SINR (dB)")

    contour_levels = np.percentile(sinr_map.reshape((-1,)), [5, 10, 20])
    ax.contour(grid_x, grid_y, sinr_map, levels=contour_levels, colors=["#7F0000", "#C43C39", "#FFB000"], linewidths=1.2)

    ax.plot(
        [p.x_aligned_m for p in env.all_points],
        [p.y_aligned_m for p in env.all_points],
        color="white",
        linewidth=0.7,
        alpha=0.18,
        label="Measured trajectory",
    )
    if dense_reference_points:
        ax.scatter(
            [p.x_aligned_m for p in dense_reference_points],
            [p.y_aligned_m for p in dense_reference_points],
            s=10,
            c="#111111",
            alpha=0.45,
            label="Measured poor-cloud",
        )
    ax.scatter(
        [p.x_aligned_m for p in env.selected_points],
        [p.y_aligned_m for p in env.selected_points],
        s=65,
        c="#FFFFFF",
        edgecolors="#000000",
        linewidths=0.8,
        label="Selected reference points",
    )
    for point in env.selected_points:
        ax.annotate(point.selection_id, (point.x_aligned_m, point.y_aligned_m), textcoords="offset points", xytext=(5, 4), fontsize=8, color="black")

    if worst_records:
        ax.scatter(
            [float(item["x_m"]) for item in worst_records],
            [float(item["y_m"]) for item in worst_records],
            s=18,
            c="#8A2BE2",
            marker="x",
            alpha=0.8,
            label="Worst-grid points",
        )

    sources = list(scenario.get("interference_sources", []))
    if sources:
        ax.scatter(
            [float(getattr(src, "x", 0.0)) for src in sources],
            [float(getattr(src, "y", 0.0)) for src in sources],
            s=26,
            c="#00E5FF",
            marker="^",
            edgecolors="#003B46",
            linewidths=0.6,
            alpha=0.75,
            label="Scenario interferers",
        )

    drones = np.asarray(scenario.get("drone_positions", np.zeros((0, 3))), dtype=float)
    if drones.size:
        ax.scatter(
            drones[:, 0],
            drones[:, 1],
            s=95,
            c="#8A2BE2",
            marker="D",
            edgecolors="black",
            linewidths=0.8,
            label="Optimized UAV position",
        )

    if env.hotspot_specs:
        ax.scatter(
            [float(item["x_m"]) for item in env.hotspot_specs],
            [float(item["y_m"]) for item in env.hotspot_specs],
            s=120,
            c="#00E5FF",
            marker="^",
            edgecolors="#003B46",
            linewidths=0.8,
            label="Aligned hotspots",
        )

    ax.scatter([env.station_xy_aligned_m[0]], [env.station_xy_aligned_m[1]], s=180, marker="*", color="black", label="Serving BS proxy")
    ax.set_xlim(0.0, float(env.area_size_m))
    ax.set_ylim(0.0, float(env.area_size_m))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("{0}-{1}: dense heatmap under optimized scene".format(env.dataset_id.upper(), group_name.upper()))
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _result_rows(points: Sequence[MeasuredPoint], records: Sequence[Mapping[str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for point, record in zip(points, records):
        row = asdict(point)
        row.update(
            {
                "model_margin_db": float(record["margin_db"]),
                "model_sinr_db": float(record["sinr_db"]),
                "model_rate_mbps": float(record["rate_bps"]) / 1e6,
                "model_outage": float(record["outage"]),
                "model_interference_dbm": float(record["i_dbm"]),
            }
        )
        rows.append(row)
    return rows


def _corner_case_report(
    bundle: DatasetBundle,
    env: EnvironmentBundle,
    group_name: str,
    group_search: Mapping[str, object],
    measured_records: Sequence[Mapping[str, float]],
    trajectory_records: Sequence[Mapping[str, float]],
    random_records: Sequence[Mapping[str, float]],
    worst_records: Sequence[Mapping[str, float]],
) -> Dict[str, object]:
    return {
        "dataset_id": bundle.dataset_id,
        "family": bundle.family,
        "reference_altitude_m": float(bundle.reference_altitude_m),
        "station_height_m": float(bundle.station_height_m),
        "comm_freq_hz": float(bundle.comm_freq_hz),
        "algorithm_group": str(group_name),
        "search": {
            "best_score": float(group_search.get("best_score", 0.0)),
            "score_stats": dict(group_search.get("score_stats", {})),
            "num_samples": int(len(group_search.get("samples", []))),
            "gan_training": dict(group_search.get("gan_training", {})) if isinstance(group_search.get("gan_training", {}), dict) else {},
        },
        "environment_alignment": {
            "notes": bundle.notes,
            "area_size_m": float(env.area_size_m),
            "station_xy_local_m": [float(env.station_xy_local_m[0]), float(env.station_xy_local_m[1])],
            "station_xy_aligned_m": [float(env.station_xy_aligned_m[0]), float(env.station_xy_aligned_m[1])],
            "shift_xy_m": [float(env.shift_xy_m[0]), float(env.shift_xy_m[1])],
            "hotspots": list(env.hotspot_specs),
        },
        "groups": {
            "measured_selected": _summarize_model_group(measured_records),
            "measured_trajectory": _summarize_model_group(trajectory_records),
            "random": _summarize_model_group(random_records),
            "model_worst_region": _summarize_model_group(worst_records),
        },
    }


def run_comparison(
    aerpaw_zip: str,
    dryad_zip: str,
    out_dir: str,
    random_points: int,
    worst_top_k: int,
    grid_size: int,
    tx_power_dbm: float,
    bandwidth_hz: float,
    use_empirical_hotspots: bool,
    seed: int,
    ga_runs: int = 40,
    ga_maxgen: int = 40,
    ga_nind: int = 30,
    gan_epochs: int = 300,
    gan_batch: int = 32,
    gan_latent: int = 32,
    gan_samples_n: int = 20,
    random_search_n: int = 40,
    device: str = "cpu",
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    bundles = (_load_aerpaw_bundle(aerpaw_zip), _load_dryad_bundle(dryad_zip))

    overall_summary: Dict[str, object] = {
        "reference_files": {
            "aerpaw_zip": aerpaw_zip,
            "dryad_zip": dryad_zip,
            "reference_pdf": DEFAULT_REF_PDF,
        },
        "datasets": {},
    }

    for offset, bundle in enumerate(bundles):
        dataset_out_dir = os.path.join(out_dir, bundle.dataset_id)
        os.makedirs(dataset_out_dir, exist_ok=True)

        env = _align_bundle(bundle, area_margin_m=60.0, use_empirical_hotspots=use_empirical_hotspots)
        problem = _build_problem(
            env=env,
            reference_altitude_m=bundle.reference_altitude_m,
            comm_freq_hz=bundle.comm_freq_hz,
            bandwidth_hz=bandwidth_hz,
            tx_power_dbm=tx_power_dbm,
        )
        hotspots = _build_hotspots(env)
        _attach_aligned_environment(problem, env, [dict(spec) for spec in env.hotspot_specs])
        model_groups = _run_original_model_groups(
            problem=problem,
            seed=int(seed + offset * 1000),
            ga_runs=int(ga_runs),
            ga_maxgen=int(ga_maxgen),
            ga_nind=int(ga_nind),
            gan_epochs=int(gan_epochs),
            gan_batch=int(gan_batch),
            gan_latent=int(gan_latent),
            gan_samples_n=int(gan_samples_n),
            random_search_n=int(random_search_n),
            device=str(device),
        )

        dense_reference_points = _extract_dense_reference_points(env.all_points, quantile=0.10)
        dataset_summary: Dict[str, object] = {
            "dataset_id": bundle.dataset_id,
            "family": bundle.family,
            "environment_alignment": {
                "notes": bundle.notes,
                "area_size_m": float(env.area_size_m),
                "station_xy_local_m": [float(env.station_xy_local_m[0]), float(env.station_xy_local_m[1])],
                "station_xy_aligned_m": [float(env.station_xy_aligned_m[0]), float(env.station_xy_aligned_m[1])],
                "shift_xy_m": [float(env.shift_xy_m[0]), float(env.shift_xy_m[1])],
                "hotspots": list(env.hotspot_specs),
            },
            "algorithm_groups": {},
        }

        for group_name, group_search in model_groups.items():
            group_out_dir = os.path.join(dataset_out_dir, group_name)
            os.makedirs(group_out_dir, exist_ok=True)
            best_x = np.asarray(group_search.get("best_x", np.zeros((problem.Dim,), dtype=float)), dtype=float)
            scenario = problem.generate_scenario(best_x)

            measured_xyz = np.asarray([[p.x_aligned_m, p.y_aligned_m, p.z_local_m] for p in env.selected_points], dtype=float)
            measured_records = _evaluate_positions(problem, scenario, measured_xyz)
            trajectory_xyz = np.asarray([[p.x_aligned_m, p.y_aligned_m, p.z_local_m] for p in env.all_points], dtype=float)
            trajectory_records = _evaluate_positions(problem, scenario, trajectory_xyz)
            random_xyz = _sample_random_positions(bundle.reference_altitude_m, env.area_size_m, random_points, seed + offset + len(group_name))
            random_records = _evaluate_positions(problem, scenario, random_xyz)
            worst_records = _search_worst_region(problem, scenario, bundle.reference_altitude_m, env.area_size_m, grid_size, worst_top_k)
            grid_x, grid_y, sinr_map, grid_records = _evaluate_grid_map(problem, scenario, bundle.reference_altitude_m, env.area_size_m, grid_size)

            _plot_spatial_layout(env, scenario, group_name, random_records, worst_records, os.path.join(group_out_dir, "{0}_{1}_spatial_compare.png".format(bundle.dataset_id, group_name)))
            _plot_dense_heatmap(
                env=env,
                scenario=scenario,
                group_name=group_name,
                dense_reference_points=dense_reference_points,
                grid_x=grid_x,
                grid_y=grid_y,
                sinr_map=sinr_map,
                worst_records=worst_records,
                out_path=os.path.join(group_out_dir, "{0}_{1}_dense_heatmap.png".format(bundle.dataset_id, group_name)),
            )
            _plot_metric_comparison(bundle.dataset_id, group_name, measured_records, random_records, worst_records, os.path.join(group_out_dir, "{0}_{1}_metric_compare.png".format(bundle.dataset_id, group_name)))

            _write_csv(os.path.join(group_out_dir, "selected_points_with_source.csv"), _result_rows(env.selected_points, measured_records))
            _write_csv(os.path.join(group_out_dir, "trajectory_points_modeled.csv"), _result_rows(env.all_points, trajectory_records))
            _write_csv(os.path.join(group_out_dir, "model_worst_region.csv"), worst_records)
            _write_csv(os.path.join(group_out_dir, "grid_map.csv"), grid_records)
            _write_csv(os.path.join(group_out_dir, "random_points.csv"), random_records)
            _write_csv(
                os.path.join(group_out_dir, "best_decision_vector.csv"),
                [{"index": int(i), "value": float(v)} for i, v in enumerate(best_x.tolist())],
            )

            dataset_report = _corner_case_report(bundle, env, group_name, group_search, measured_records, trajectory_records, random_records, worst_records)
            with open(os.path.join(group_out_dir, "corner_case_report.json"), "w", encoding="utf-8") as handle:
                json.dump(dataset_report, handle, ensure_ascii=False, indent=2)
            dataset_summary["algorithm_groups"][group_name] = dataset_report

        overall_summary["datasets"][bundle.dataset_id] = dataset_summary

    with open(os.path.join(out_dir, "corner_case_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, ensure_ascii=False, indent=2)
    return overall_summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare measured trajectory corner cases against model worst region and random points.")
    parser.add_argument("--aerpaw_zip", default=DEFAULT_AERPAW_ZIP, help="Path to aerpaw-dataset-24.zip")
    parser.add_argument("--dryad_zip", default=DEFAULT_DRYAD_ZIP, help="Path to doi_10_5061_dryad_wh70rxx06__v20250521.zip")
    parser.add_argument("--out_dir", default=os.path.join("output", "measured_corner_compare"), help="Output directory")
    parser.add_argument("--random_points", type=int, default=120, help="Random reference points per dataset")
    parser.add_argument("--worst_top_k", type=int, default=25, help="How many grid points to retain as model worst region")
    parser.add_argument("--grid_size", type=int, default=55, help="Grid resolution per axis for worst-region search")
    parser.add_argument("--tx_power_dbm", type=float, default=43.0, help="Serving BS transmit power used by the model")
    parser.add_argument("--bandwidth_hz", type=float, default=100e6, help="Bandwidth used for Shannon-rate evaluation")
    parser.add_argument("--seed", type=int, default=20260310, help="Random seed")
    parser.add_argument("--disable_hotspots", action="store_true", help="Disable empirical hotspot alignment derived from measured poor points")
    parser.add_argument("--ga_runs", type=int, default=40)
    parser.add_argument("--ga_maxgen", type=int, default=40)
    parser.add_argument("--ga_nind", type=int, default=30)
    parser.add_argument("--gan_epochs", type=int, default=300)
    parser.add_argument("--gan_batch", type=int, default=32)
    parser.add_argument("--gan_latent", type=int, default=32)
    parser.add_argument("--gan_samples_n", type=int, default=20)
    parser.add_argument("--random_search_n", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    summary = run_comparison(
        aerpaw_zip=args.aerpaw_zip,
        dryad_zip=args.dryad_zip,
        out_dir=args.out_dir,
        random_points=args.random_points,
        worst_top_k=args.worst_top_k,
        grid_size=args.grid_size,
        tx_power_dbm=args.tx_power_dbm,
        bandwidth_hz=args.bandwidth_hz,
        use_empirical_hotspots=not bool(args.disable_hotspots),
        seed=args.seed,
        ga_runs=args.ga_runs,
        ga_maxgen=args.ga_maxgen,
        ga_nind=args.ga_nind,
        gan_epochs=args.gan_epochs,
        gan_batch=args.gan_batch,
        gan_latent=args.gan_latent,
        gan_samples_n=args.gan_samples_n,
        random_search_n=args.random_search_n,
        device=args.device,
    )
    print(json.dumps({"out_dir": args.out_dir, "datasets": list(summary.get("datasets", {}).keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
