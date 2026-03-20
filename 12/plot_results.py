from __future__ import annotations

import os
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _group_names(report: Mapping) -> Sequence[str]:
    groups = report.get("groups", {}) if isinstance(report, dict) else {}
    return [k for k in groups.keys()]


def _get_outage(report: Mapping, group: str) -> Tuple[np.ndarray, np.ndarray]:
    groups = report["groups"]
    outage = groups[group]["kpis"]["outage"]
    xs = np.asarray([float(k) for k in outage.keys()], dtype=float)
    ys = np.asarray([float(outage[str(k)]) for k in outage.keys()], dtype=float)
    order = np.argsort(xs)
    return xs[order], ys[order]


def _get_stat(report: Mapping, group: str, key: str) -> Dict[str, float]:
    return dict(report["groups"][group]["kpis"][key])


def _get_thr_stat(report: Mapping, group: str, which: str) -> Dict[str, float]:
    return dict(report["groups"][group][which]["throughput_bps"])


def _bar_with_quantiles(ax, groups: Sequence[str], stats: Dict[str, Dict[str, float]], title: str, ylabel: str):
    means = [float(stats[g].get("mean", 0.0)) for g in groups]
    p10 = [float(stats[g].get("p10", means[i])) for i, g in enumerate(groups)]
    p90 = [float(stats[g].get("p90", means[i])) for i, g in enumerate(groups)]
    x = np.arange(len(groups))
    ax.bar(x, means, color=["#4ECDC4", "#FF6B6B", "#95A5A6"][: len(groups)], alpha=0.8)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(p10), np.array(p90) - np.array(means)], fmt="none", ecolor="k", capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([g.upper() for g in groups])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")


def plot_kpi_comparison(report: Mapping, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    groups = _group_names(report)
    if not groups:
        return

    # 1) Outage curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in groups:
        xs, ys = _get_outage(report, g)
        ax.plot(xs, ys, marker="o", linewidth=2, label=g.upper())
    ax.set_xlabel("Threshold θ (dB)")
    ax.set_ylabel("P_out(θ) = Pr(SINR < θ)")
    ax.set_title("Outage Probability (multi-threshold)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kpi_outage_curve.png"), dpi=150)
    plt.close(fig)

    # 2) Rate stats
    rate_stats = {g: _get_stat(report, g, "rate_bps") for g in groups}
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_with_quantiles(ax, groups, rate_stats, "Rate (Shannon upper bound)", "bps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kpi_rate_stats.png"), dpi=150)
    plt.close(fig)

    # 3) EE stats
    ee_stats = {g: _get_stat(report, g, "ee_bpj") for g in groups}
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_with_quantiles(ax, groups, ee_stats, "Energy Efficiency (bit/J)", "bit/J")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kpi_ee_stats.png"), dpi=150)
    plt.close(fig)

    # 4) Tail risk (worst 5%)
    tail_sinr = {g: float(report["groups"][g]["kpis"]["tail"]["tail_mean_sinr_db"]) for g in groups}
    tail_rate = {g: float(report["groups"][g]["kpis"]["tail"]["tail_mean_rate_bps"]) for g in groups}
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, [tail_sinr[g] for g in groups], width=width, label="Worst 5% mean SINR (dB)")
    ax.bar(x + width / 2, [tail_rate[g] / 1e6 for g in groups], width=width, label="Worst 5% mean Rate (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels([g.upper() for g in groups])
    ax.set_title("Tail Risk (worst 5%)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kpi_tail_risk.png"), dpi=150)
    plt.close(fig)

    # 5) Throughput A
    if all("throughput_A" in report["groups"][g] for g in groups):
        thr_a = {g: _get_thr_stat(report, g, "throughput_A") for g in groups}
        fig, ax = plt.subplots(figsize=(6, 4))
        _bar_with_quantiles(ax, groups, thr_a, "Throughput (BLER_A)", "bps")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "kpi_throughput_A.png"), dpi=150)
        plt.close(fig)

    # 6) Throughput B (optional)
    if all("throughput_B" in report["groups"][g] for g in groups):
        thr_b = {g: _get_thr_stat(report, g, "throughput_B") for g in groups}
        fig, ax = plt.subplots(figsize=(6, 4))
        _bar_with_quantiles(ax, groups, thr_b, "Throughput (BLER_B / Sionna LUT)", "bps")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "kpi_throughput_B.png"), dpi=150)
        plt.close(fig)

