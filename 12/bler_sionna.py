from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SionnaConfig:
    modulation: str = "qpsk"  # qpsk|16qam
    n_bits_per_block: int = 1024
    n_blocks: int = 400
    seed: int = 0
    channel: str = "awgn"  # awgn|rician
    rician_k: float = 5.0


def _require_sionna():
    try:
        import tensorflow as tf  # noqa: F401
        import sionna  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Sionna/TensorFlow not available. Install tensorflow + sionna first, "
            "or provide an existing BLER_B.csv LUT."
        ) from e


def simulate_bler_curve_sionna(
    sinr_db_grid: Iterable[float],
    cfg: SionnaConfig = SionnaConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    _require_sionna()

    import tensorflow as tf
    # Sionna v1.x namespace
    from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper
    from sionna.phy.channel import AWGN

    mod = str(cfg.modulation).strip().lower()
    if mod not in ("qpsk", "16qam"):
        raise ValueError("modulation must be 'qpsk' or '16qam'")

    bits_per_sym = 2 if mod == "qpsk" else 4
    n_bits = int(max(cfg.n_bits_per_block, bits_per_sym))
    n_bits = int((n_bits // bits_per_sym) * bits_per_sym)
    n_syms = n_bits // bits_per_sym
    n_blocks = int(max(cfg.n_blocks, 1))

    tf.random.set_seed(int(cfg.seed))
    bs = BinarySource()

    constellation = Constellation("qam", num_bits_per_symbol=int(bits_per_sym))
    mapper = Mapper(constellation=constellation)
    # Demapper returns LLRs by default; use hard_out=True for hard bits {0,1}
    demapper = Demapper("maxlog", constellation=constellation, hard_out=True)
    awgn = AWGN()

    use_rician = str(cfg.channel).strip().lower() in ("rician", "flat_rician", "flatfading")
    if use_rician:
        # Flat fading with Rician K-factor (approx): LoS + scatter
        # Sionna FlatFadingChannel uses complex Gaussian fading; we emulate Rician via mean+variance
        k = float(max(cfg.rician_k, 0.0))
        mu = np.sqrt(k / (k + 1.0))
        sigma = np.sqrt(1.0 / (k + 1.0))

        def sample_h(batch: int) -> tf.Tensor:
            re = tf.random.normal([batch, 1], mean=mu, stddev=sigma, dtype=tf.float32)
            im = tf.random.normal([batch, 1], mean=0.0, stddev=sigma, dtype=tf.float32)
            return tf.complex(re, im)

    sinr_db_arr = np.asarray(list(sinr_db_grid), dtype=float).reshape((-1,))
    bler_arr = np.zeros_like(sinr_db_arr, dtype=float)

    for i, snr_db in enumerate(sinr_db_arr):
        snr_lin = float(10.0 ** (float(snr_db) / 10.0))
        noise_var = float(1.0 / max(snr_lin, 1e-12))
        block_err = 0

        for _ in range(n_blocks):
            b = bs([1, n_bits])
            x = mapper(b)
            # x: [1, n_syms]
            if use_rician:
                h = sample_h(1)
                y = x * h
                y = awgn(y, tf.cast(noise_var, tf.float32))
                y = y / h
                no_eff = tf.cast(noise_var, tf.float32) / tf.cast(tf.abs(h) ** 2, tf.float32)
            else:
                y = awgn(x, tf.cast(noise_var, tf.float32))
                no_eff = tf.cast(noise_var, tf.float32)

            b_hat = demapper(y, no_eff)
            bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(b, b_hat), tf.int32)).numpy().item()
            if int(bit_errors) > 0:
                block_err += 1

        bler_arr[i] = float(block_err / n_blocks)

    return sinr_db_arr, np.clip(bler_arr, 0.0, 1.0)


def save_bler_csv(path: str, sinr_db: np.ndarray, bler: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sinr_db", "bler"])
        for x, y in zip(sinr_db.tolist(), bler.tolist()):
            w.writerow([float(x), float(y)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="BLER_B.csv", help="output CSV path")
    ap.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam"])
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", default="awgn", choices=["awgn", "rician"])
    ap.add_argument("--rician_k", type=float, default=5.0)
    ap.add_argument("--sinr_db", default="-5,-2,0,2,4,6,8,10,12,15", help="comma-separated SINR(dB) points")
    args = ap.parse_args()

    grid = [float(x.strip()) for x in str(args.sinr_db).split(",") if x.strip()]
    cfg = SionnaConfig(
        modulation=args.mod,
        n_bits_per_block=args.bits,
        n_blocks=args.blocks,
        seed=args.seed,
        channel=args.channel,
        rician_k=float(args.rician_k),
    )
    sinr_db, bler = simulate_bler_curve_sionna(grid, cfg=cfg)
    save_bler_csv(str(args.out), sinr_db, bler)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
