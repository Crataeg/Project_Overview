import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from UAV_GA import DroneCommProblem, save_scenario_data


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


def scale_to_unit(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    denom = np.maximum(ub - lb, 1e-9)
    return 2.0 * (x - lb) / denom - 1.0


def scale_from_unit(z: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return 0.5 * (z + 1.0) * (ub - lb) + lb


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


def run_ga_samples(problem: DroneCommProblem, runs: int, maxgen: int,
                   nind: int, seed_base: int) -> np.ndarray:
    try:
        import geatpy as ea
    except Exception as exc:
        raise RuntimeError("geatpy is required to run GA sampling") from exc

    samples: List[np.ndarray] = []
    for i in range(int(runs)):
        np.random.seed(int(seed_base) + i)
        pop = ea.Population(Encoding="RI", NIND=int(nind))
        algorithm = ea.soea_SEGA_templet(
            problem,
            pop,
            MAXGEN=int(maxgen),
            logTras=0,
            trappedValue=1e-4,
            maxTrappedCount=10
        )
        res = ea.optimize(
            algorithm,
            verbose=False,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )
        best_x = np.asarray(res["Vars"][0], dtype=float)
        samples.append(best_x)
    return np.asarray(samples, dtype=float)


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_gan(samples: np.ndarray, lb: np.ndarray, ub: np.ndarray,
              epochs: int, batch_size: int, latent_dim: int,
              device: str) -> Tuple[MLPGenerator, Dict]:
    if torch is None:
        raise RuntimeError("torch is required to train the GAN")

    x_unit = scale_to_unit(samples, lb, ub).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x_unit))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=True)

    gen = MLPGenerator(latent_dim, x_unit.shape[1]).to(device)
    disc = MLPDiscriminator(x_unit.shape[1]).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    log = {"epochs": int(epochs), "batch_size": int(batch_size), "latent_dim": int(latent_dim)}

    for _ in range(int(epochs)):
        for batch in loader:
            real = batch[0].to(device)
            bsz = real.size(0)

            z = torch.randn(bsz, latent_dim, device=device)
            fake = gen(z).detach()

            opt_d.zero_grad()
            pred_real = disc(real)
            pred_fake = disc(fake)
            loss_d = criterion(pred_real, torch.ones_like(pred_real)) + \
                     criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_d.backward()
            opt_d.step()

            z = torch.randn(bsz, latent_dim, device=device)
            fake = gen(z)
            opt_g.zero_grad()
            pred = disc(fake)
            loss_g = criterion(pred, torch.ones_like(pred))
            loss_g.backward()
            opt_g.step()

    return gen, log


def generate_gan_samples(gen: MLPGenerator, n_samples: int, latent_dim: int,
                         lb: np.ndarray, ub: np.ndarray,
                         type_idx: np.ndarray, loc_idx: np.ndarray,
                         device: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to generate GAN samples")
    gen.eval()
    with torch.no_grad():
        z = torch.randn(int(n_samples), int(latent_dim), device=device)
        x_unit = gen(z).cpu().numpy()
    x = scale_from_unit(x_unit, lb, ub)
    return postprocess_x(x, lb, ub, type_idx, loc_idx)


def summarize_metrics(metrics: List[Dict]) -> Dict:
    def _stat(values: List[float]) -> Dict:
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

    keys = [
        "avg_total_deg", "avg_comm_deg", "avg_cdi",
        "avg_interference_dbm", "avg_margin_db"
    ]
    summary = {}
    for key in keys:
        summary[key] = _stat([float(m.get(key, 0.0)) for m in metrics])
    return summary


def plot_gan_metrics(metrics: List[Dict], out_dir: str) -> None:
    keys = [
        "avg_total_deg", "avg_comm_deg", "avg_cdi",
        "avg_interference_dbm", "avg_margin_db"
    ]
    data = [[float(m.get(key, 0.0)) for m in metrics] for key in keys]
    if not data or all(len(v) == 0 for v in data):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=keys, showfliers=False)
    ax.set_title("GAN Scenario Metrics Distribution")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gan_metrics_box.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if data and data[0]:
        ax.hist(data[0], bins=15, color="#4C78A8", alpha=0.85)
        ax.set_title("GAN avg_total_deg Histogram")
        ax.set_xlabel("avg_total_deg")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gan_avg_total_deg_hist.png"), dpi=150)
    plt.close(fig)


def plot_gan_feature_distributions(height_ratios: List[float],
                                   speeds: List[float],
                                   out_dir: str) -> None:
    if not height_ratios and not speeds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if height_ratios:
        axes[0].hist(height_ratios, bins=20, color="#F58518", alpha=0.85)
        axes[0].set_title("UAV Height / Mean Building Height")
        axes[0].set_xlabel("Height Ratio")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0].set_title("UAV Height Ratio (No Buildings)")
        axes[0].axis("off")

    if speeds:
        axes[1].hist(speeds, bins=20, color="#54A24B", alpha=0.85)
        axes[1].set_title("UAV Speed Distribution")
        axes[1].set_xlabel("Speed (m/s)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].set_title("UAV Speed Distribution (Empty)")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gan_uav_height_speed.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="", help="path to JSON with user_params overrides")
    parser.add_argument("--ga_runs", type=int, default=40)
    parser.add_argument("--ga_maxgen", type=int, default=40)
    parser.add_argument("--ga_nind", type=int, default=30)
    parser.add_argument("--ga_seed", type=int, default=123)
    parser.add_argument("--gan_epochs", type=int, default=300)
    parser.add_argument("--gan_batch", type=int, default=32)
    parser.add_argument("--gan_latent", type=int, default=32)
    parser.add_argument("--gan_samples", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run_id", default="", help="output run folder id")
    args = parser.parse_args()

    run_id = args.run_id.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = os.path.join("output", run_id)
    base_output = os.path.join(run_root, "gan")
    arrays_output = os.path.join(base_output, "arrays")
    scenario_output = os.path.join(base_output, "scenarios")
    viz_output = os.path.join(base_output, "visuals")
    os.makedirs(arrays_output, exist_ok=True)
    os.makedirs(scenario_output, exist_ok=True)
    os.makedirs(viz_output, exist_ok=True)

    user_params = load_user_params(args.params)
    problem = DroneCommProblem(user_params)

    ga_samples = run_ga_samples(problem, args.ga_runs, args.ga_maxgen, args.ga_nind, args.ga_seed)
    np.save(os.path.join(arrays_output, "ga_samples.npy"), ga_samples)

    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    type_idx, loc_idx = discrete_indices(problem)

    if torch is None:
        raise RuntimeError("torch is required to train the GAN; install pytorch first")
    device = str(args.device)

    gen, gan_log = train_gan(
        ga_samples, lb, ub, args.gan_epochs, args.gan_batch, args.gan_latent, device
    )

    gan_samples = generate_gan_samples(
        gen, args.gan_samples, args.gan_latent, lb, ub, type_idx, loc_idx, device
    )
    np.save(os.path.join(arrays_output, "gan_samples.npy"), gan_samples)

    metrics = []
    height_ratios: List[float] = []
    speed_values: List[float] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for i, x in enumerate(gan_samples):
        scenario = problem.generate_scenario(np.asarray(x, dtype=float))
        try:
            scenario["analysis"] = problem.analyze_scenario_metrics(scenario)
        except Exception:
            scenario["analysis"] = {}
        fname = os.path.join(run_id, "gan", "scenarios", f"gan_scenario_{timestamp}_{i:03d}.json")
        save_scenario_data(scenario, fname)
        metrics.append(scenario.get("analysis", {}))

        speeds = scenario.get("drone_speeds", [])
        for v in speeds:
            speed_values.append(float(v))
        buildings = scenario.get("buildings", [])
        if buildings:
            heights = [float(b.height) for b in buildings]
            mean_h = float(np.mean(heights)) if heights else 0.0
            if mean_h > 0.0:
                drone_h = scenario.get("drone_positions", [])[:, 2]
                for h in drone_h:
                    height_ratios.append(float(h) / mean_h)

    summary = summarize_metrics(metrics)
    summary["gan_training"] = gan_log
    with open(os.path.join(base_output, "gan_distribution_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_gan_metrics(metrics, viz_output)
    plot_gan_feature_distributions(height_ratios, speed_values, viz_output)

    print(f"Run folder: output/{run_id}/gan")
    print("GA samples saved:", os.path.join(base_output, "arrays", "ga_samples.npy"))
    print("GAN samples saved:", os.path.join(base_output, "arrays", "gan_samples.npy"))
    print("GAN scenarios saved:", os.path.join(base_output, "scenarios", "gan_scenario_*.json"))
    print("GAN distribution summary:", os.path.join(base_output, "gan_distribution_summary.json"))
    print("GAN visualizations:",
          os.path.join(base_output, "visuals", "gan_metrics_box.png"),
          os.path.join(base_output, "visuals", "gan_avg_total_deg_hist.png"),
          os.path.join(base_output, "visuals", "gan_uav_height_speed.png"))


if __name__ == "__main__":
    main()
