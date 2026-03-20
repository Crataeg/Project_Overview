import os
import sys
import subprocess
import time


def prompt_str(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def prompt_int(prompt: str, default: int) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return int(default)
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return float(default)
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def script_path(name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, name)


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_run_id(prefix: str = "run") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S")


def collect_params_path() -> str:
    path = prompt_str("Path to params JSON (optional)", "")
    if path and not os.path.isfile(path):
        print("File not found, using default params.")
        return ""
    return path


def collect_ga_settings() -> dict:
    return {
        "ga_runs": prompt_int("GA runs", 40),
        "ga_maxgen": prompt_int("GA MAXGEN", 40),
        "ga_nind": prompt_int("GA population size (NIND)", 30),
        "ga_seed": prompt_int("GA random seed", 123),
    }


def collect_gan_settings() -> dict:
    return {
        "gan_epochs": prompt_int("GAN epochs", 300),
        "gan_batch": prompt_int("GAN batch size", 32),
        "gan_latent": prompt_int("GAN latent dim", 32),
        "gan_samples": prompt_int("GAN samples to generate", 20),
        "device": prompt_str("Torch device (cpu/cuda)", "cpu"),
    }


def collect_random_settings() -> dict:
    return {
        "random_n": prompt_int("Random samples", 50),
        "random_seed": prompt_int("Random seed", 2024),
    }


def run_gan_pipeline(params_path: str, ga_cfg: dict, gan_cfg: dict, run_id: str):
    cmd = [sys.executable, script_path("gan_uav_pipeline.py")]
    if params_path:
        cmd += ["--params", params_path]
    cmd += [
        "--ga_runs", str(ga_cfg["ga_runs"]),
        "--ga_maxgen", str(ga_cfg["ga_maxgen"]),
        "--ga_nind", str(ga_cfg["ga_nind"]),
        "--ga_seed", str(ga_cfg["ga_seed"]),
        "--gan_epochs", str(gan_cfg["gan_epochs"]),
        "--gan_batch", str(gan_cfg["gan_batch"]),
        "--gan_latent", str(gan_cfg["gan_latent"]),
        "--gan_samples", str(gan_cfg["gan_samples"]),
        "--device", str(gan_cfg["device"]),
        "--run_id", str(run_id),
    ]
    run_command(cmd)


def run_comparison(params_path: str, random_cfg: dict, run_id: str = ""):
    cmd = [sys.executable, script_path("compare_random_ga_gan.py")]
    if params_path:
        cmd += ["--params", params_path]
    if run_id:
        cmd += ["--run_id", run_id]
    cmd += [
        "--random_n", str(random_cfg["random_n"]),
        "--random_seed", str(random_cfg["random_seed"]),
    ]
    run_command(cmd)


def main():
    print("UAV GA + GAN interactive CLI")
    print("Output files will be written under ./output")

    while True:
        print("")
        print("1) Train GAN and generate scenarios")
        print("2) Compare GA/GAN/Random")
        print("3) Full pipeline (1 then 2)")
        print("4) Quick demo (small GA/GAN/compare)")
        print("5) Exit")
        choice = prompt_str("Select option", "")

        if choice == "1":
            params_path = collect_params_path()
            ga_cfg = collect_ga_settings()
            gan_cfg = collect_gan_settings()
            run_id = make_run_id("run")
            run_gan_pipeline(params_path, ga_cfg, gan_cfg, run_id)
        elif choice == "2":
            params_path = collect_params_path()
            random_cfg = collect_random_settings()
            run_id = prompt_str("Run id (empty = latest)", "")
            run_comparison(params_path, random_cfg, run_id)
        elif choice == "3":
            params_path = collect_params_path()
            ga_cfg = collect_ga_settings()
            gan_cfg = collect_gan_settings()
            random_cfg = collect_random_settings()
            run_id = make_run_id("run")
            run_gan_pipeline(params_path, ga_cfg, gan_cfg, run_id)
            run_comparison(params_path, random_cfg, run_id)
        elif choice == "4":
            params_path = collect_params_path()
            run_id = make_run_id("demo")
            cmd = [sys.executable, script_path("quick_demo_pipeline.py"), "--run_id", run_id]
            if params_path:
                cmd += ["--params", params_path]
            run_command(cmd)
        elif choice == "5":
            print("Done.")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
