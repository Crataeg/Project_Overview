import argparse
import os
import subprocess
import sys
import time


def script_path(name: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, name)


def run_command(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="", help="path to JSON with user_params overrides")
    parser.add_argument("--run_id", default="", help="output run folder id")
    args = parser.parse_args()

    run_id = args.run_id.strip() or time.strftime("demo_%Y%m%d_%H%M%S")

    gan_cmd = [
        sys.executable, script_path("gan_uav_pipeline.py"),
        "--run_id", run_id,
        "--ga_runs", "2",
        "--ga_maxgen", "2",
        "--ga_nind", "10",
        "--gan_epochs", "2",
        "--gan_samples", "2",
        "--gan_batch", "8",
        "--gan_latent", "16",
    ]
    if args.params:
        gan_cmd += ["--params", args.params]

    cmp_cmd = [
        sys.executable, script_path("compare_random_ga_gan.py"),
        "--run_id", run_id,
        "--random_n", "10",
        "--viz_count", "2",
    ]
    if args.params:
        cmp_cmd += ["--params", args.params]

    run_command(gan_cmd)
    run_command(cmp_cmd)

    print(f"Quick demo finished. Output: output/{run_id}/")


if __name__ == "__main__":
    main()
