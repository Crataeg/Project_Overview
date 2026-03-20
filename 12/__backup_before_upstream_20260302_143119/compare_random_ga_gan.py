import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from UAV_GA import DroneCommProblem, EnhancedVisualizer


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
        "num_controlled_interference": 8
    }


def load_user_params(path: str) -> Dict:
    params = default_user_params()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    return params


def discrete_indices(problem: DroneCommProblem) -> Tuple[np.ndarray, np.ndarray]:
    d = int(problem.num_drones)
    m = int(problem.num_interference)
    type_idx = np.arange(4 * d, 4 * d + m)
    loc_idx = np.arange(4 * d + 2 * m, 4 * d + 3 * m)
    return type_idx, loc_idx


def postprocess_x(x: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                  type_idx: np.ndarray, loc_idx: np.ndarray) -> np.ndarray:
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


def evaluate_samples(problem: DroneCommProblem, samples: np.ndarray) -> List[Dict]:
    metrics = []
    for x in samples:
        scenario = problem.generate_scenario(np.asarray(x, dtype=float))
        try:
            scenario["analysis"] = problem.analyze_scenario_metrics(scenario)
        except Exception:
            scenario["analysis"] = {}
        metrics.append(scenario["analysis"])
    return metrics


def summarize(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def summarize_metrics(metrics: List[Dict]) -> Dict:
    keys = [
        "avg_total_deg", "avg_comm_deg", "avg_cdi",
        "avg_interference_dbm", "avg_margin_db"
    ]
    out = {}
    for key in keys:
        out[key] = summarize([float(m.get(key, 0.0)) for m in metrics])
    return out


def diff_stats(a: Dict, b: Dict) -> Dict:
    out = {}
    for key, stats in a.items():
        if key not in b or "mean" not in stats or "mean" not in b[key]:
            continue
        out[key] = {
            "mean_diff": float(stats["mean"] - b[key]["mean"]),
            "mean_ratio": float(stats["mean"] / max(b[key]["mean"], 1e-9)),
            "p90_diff": float(stats.get("p90", 0.0) - b[key].get("p90", 0.0))
        }
    return out


def load_samples(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def find_latest_run_id() -> str:
    if not os.path.isdir("output"):
        return ""
    candidates = []
    for name in os.listdir("output"):
        full = os.path.join("output", name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("run_"):
            continue
        if os.path.isdir(os.path.join(full, "gan")):
            candidates.append(full)
    if not candidates:
        return ""
    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    return os.path.basename(latest)


def plot_comparison(ga_metrics: List[Dict], gan_metrics: List[Dict],
                    rnd_metrics: List[Dict], out_dir: str) -> None:
    keys = [
        "avg_total_deg", "avg_comm_deg", "avg_cdi",
        "avg_interference_dbm", "avg_margin_db"
    ]
    labels = ["GA", "GAN", "Random"]

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        data = [
            [float(m.get(key, 0.0)) for m in ga_metrics],
            [float(m.get(key, 0.0)) for m in gan_metrics],
            [float(m.get(key, 0.0)) for m in rnd_metrics],
        ]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(key)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("GA vs GAN vs Random")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_box.png"), dpi=150)
    plt.close(fig)

    means = {
        "GA": [float(np.mean([float(m.get(k, 0.0)) for m in ga_metrics])) for k in keys],
        "GAN": [float(np.mean([float(m.get(k, 0.0)) for m in gan_metrics])) for k in keys],
        "Random": [float(np.mean([float(m.get(k, 0.0)) for m in rnd_metrics])) for k in keys],
    }
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, means["GA"], width, label="GA")
    ax.bar(x, means["GAN"], width, label="GAN")
    ax.bar(x + width, means["Random"], width, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Mean")
    ax.set_title("Mean Metrics Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_means.png"), dpi=150)
    plt.close(fig)


def plot_total_deg_focus(ga_metrics: List[Dict], gan_metrics: List[Dict],
                         rnd_metrics: List[Dict], out_dir: str) -> None:
    key = "avg_total_deg"
    labels = ["GA", "GAN", "Random"]
    data = [
        [float(m.get(key, 0.0)) for m in ga_metrics],
        [float(m.get(key, 0.0)) for m in gan_metrics],
        [float(m.get(key, 0.0)) for m in rnd_metrics],
    ]
    if all(len(v) == 0 for v in data):
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ga_mean = float(np.mean(data[0])) if data[0] else 0.0
    gan_mean = float(np.mean(data[1])) if data[1] else 0.0
    ax.scatter([1], [ga_mean], color="#D62728", zorder=3, marker="D", label="GA mean")
    ax.scatter([2], [gan_mean], color="#9467BD", zorder=3, marker="D", label="GAN mean")
    ax.text(1, ga_mean, f"{ga_mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(2, gan_mean, f"{gan_mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Comprehensive Degradation (avg_total_deg)")
    ax.set_ylabel("avg_total_deg")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02, "Metric from GA degradation model", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_total_deg_overview.png"), dpi=150)
    plt.close(fig)

    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in data if v])
    if all_vals.size == 0:
        return
    bins = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), 20)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in zip(labels, data):
        if not values:
            continue
        hist, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, label=label, linewidth=2)
    ax.set_title("Overall Degradation Density")
    ax.set_xlabel("avg_total_deg")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_total_deg_density.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="", help="path to JSON with user_params overrides")
    parser.add_argument("--ga_samples", default="")
    parser.add_argument("--gan_samples", default="")
    parser.add_argument("--run_id", default="", help="output run folder id")
    parser.add_argument("--viz_count", type=int, default=5)
    parser.add_argument("--random_n", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=2024)
    args = parser.parse_args()

    run_id = args.run_id.strip() or find_latest_run_id()
    if not run_id:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")

    base_output = os.path.join("output", run_id, "compare")
    visuals_output = os.path.join(base_output, "visuals")
    ga_visuals = os.path.join(visuals_output, "ga")
    gan_visuals = os.path.join(visuals_output, "gan")
    rnd_visuals = os.path.join(visuals_output, "random")
    os.makedirs(ga_visuals, exist_ok=True)
    os.makedirs(gan_visuals, exist_ok=True)
    os.makedirs(rnd_visuals, exist_ok=True)

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

    ga_metrics = evaluate_samples(problem, ga_samples)
    gan_metrics = evaluate_samples(problem, gan_samples)
    rnd_metrics = evaluate_samples(problem, rnd_samples)

    ga_summary = summarize_metrics(ga_metrics)
    gan_summary = summarize_metrics(gan_metrics)
    rnd_summary = summarize_metrics(rnd_metrics)

    comparison = {
        "ga_vs_random": diff_stats(ga_summary, rnd_summary),
        "gan_vs_random": diff_stats(gan_summary, rnd_summary),
        "ga_vs_gan": diff_stats(ga_summary, gan_summary)
    }

    report = {
        "ga_summary": ga_summary,
        "gan_summary": gan_summary,
        "random_summary": rnd_summary,
        "comparison": comparison
    }

    with open(os.path.join(base_output, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_comparison(ga_metrics, gan_metrics, rnd_metrics, base_output)
    plot_total_deg_focus(ga_metrics, gan_metrics, rnd_metrics, base_output)

    viz_count = int(max(args.viz_count, 0))
    if viz_count > 0:
        viz_count = min(viz_count, len(ga_samples), len(gan_samples), len(rnd_samples))

        def visualize_samples(samples: np.ndarray, out_dir: str, tag: str):
            for i in range(viz_count):
                scenario = problem.generate_scenario(np.asarray(samples[i], dtype=float))
                vis = EnhancedVisualizer(scenario)
                prefix = f"{tag}_{run_id}_{i:03d}"
                plots = vis.create_individual_plots(prefix=prefix)
                for _, path in plots:
                    if os.path.isfile(path):
                        target = os.path.join(out_dir, os.path.basename(path))
                        shutil.move(path, target)

        visualize_samples(ga_samples, ga_visuals, "ga")
        visualize_samples(gan_samples, gan_visuals, "gan")
        visualize_samples(rnd_samples, rnd_visuals, "random")

    print(f"Run folder: output/{run_id}/compare")
    print("Comparison summary:", os.path.join(base_output, "comparison_summary.json"))
    print("Comparison visualizations:",
          os.path.join(base_output, "comparison_box.png"),
          os.path.join(base_output, "comparison_means.png"),
          os.path.join(base_output, "comparison_total_deg_overview.png"),
          os.path.join(base_output, "comparison_total_deg_density.png"))
    if viz_count > 0:
        print("Scenario visualizations:",
              os.path.join(visuals_output, "ga"),
              os.path.join(visuals_output, "gan"),
              os.path.join(visuals_output, "random"))


if __name__ == "__main__":
    main()
