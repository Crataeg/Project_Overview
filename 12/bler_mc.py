from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


def _qpsk_map(bits: np.ndarray) -> np.ndarray:
    # bits shape: (N, 2) -> symbols shape: (N,)
    b0 = bits[:, 0].astype(np.float32)
    b1 = bits[:, 1].astype(np.float32)
    # Gray mapping: 00->(1+1j), 01->(1-1j), 11->(-1-1j), 10->(-1+1j)
    i = 1.0 - 2.0 * b0
    q = 1.0 - 2.0 * b1
    s = (i + 1j * q) / np.sqrt(2.0)
    return s.astype(np.complex64)


def _qpsk_demod_hard(symbols: np.ndarray) -> np.ndarray:
    i = (np.real(symbols) < 0.0).astype(np.int8)
    q = (np.imag(symbols) < 0.0).astype(np.int8)
    return np.stack([i, q], axis=1)


def _qam16_map(bits: np.ndarray) -> np.ndarray:
    # bits shape: (N, 4) Gray mapping with levels {-3,-1,1,3}, normalized
    b0, b1, b2, b3 = [bits[:, k].astype(np.int8) for k in range(4)]

    def gray_2bit_to_level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        # 00->-3, 01->-1, 11->+1, 10->+3
        lvl = np.empty_like(msb, dtype=np.float32)
        m0 = (msb == 0) & (lsb == 0)
        m1 = (msb == 0) & (lsb == 1)
        m2 = (msb == 1) & (lsb == 1)
        m3 = (msb == 1) & (lsb == 0)
        lvl[m0] = -3.0
        lvl[m1] = -1.0
        lvl[m2] = 1.0
        lvl[m3] = 3.0
        return lvl

    i = gray_2bit_to_level(b0, b1)
    q = gray_2bit_to_level(b2, b3)
    s = (i + 1j * q) / np.sqrt(10.0)  # average symbol energy = 1
    return s.astype(np.complex64)


def _qam16_demod_hard(symbols: np.ndarray) -> np.ndarray:
    # De-normalize
    x = symbols * np.sqrt(10.0)
    i = np.real(x)
    q = np.imag(x)

    def level_to_gray_bits(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # decision boundaries at -2, 0, 2
        # levels: -3,-1,1,3 => bits: 00,01,11,10
        msb = np.zeros_like(v, dtype=np.int8)
        lsb = np.zeros_like(v, dtype=np.int8)
        m0 = v < -2.0
        m1 = (v >= -2.0) & (v < 0.0)
        m2 = (v >= 0.0) & (v < 2.0)
        m3 = v >= 2.0
        # 00
        msb[m0] = 0
        lsb[m0] = 0
        # 01
        msb[m1] = 0
        lsb[m1] = 1
        # 11
        msb[m2] = 1
        lsb[m2] = 1
        # 10
        msb[m3] = 1
        lsb[m3] = 0
        return msb, lsb

    b0, b1 = level_to_gray_bits(i)
    b2, b3 = level_to_gray_bits(q)
    return np.stack([b0, b1, b2, b3], axis=1)


def _awgn(symbols: np.ndarray, snr_linear: float, rng: np.random.Generator) -> np.ndarray:
    snr_linear = float(max(snr_linear, 1e-12))
    noise_var = 1.0 / snr_linear
    n = (rng.normal(scale=np.sqrt(noise_var / 2.0), size=symbols.shape) +
         1j * rng.normal(scale=np.sqrt(noise_var / 2.0), size=symbols.shape)).astype(np.complex64)
    return symbols + n


@dataclass(frozen=True)
class McConfig:
    modulation: str = "qpsk"  # qpsk|16qam
    n_bits_per_block: int = 1024
    n_blocks: int = 400
    seed: int = 0


def simulate_bler_curve(
    sinr_db_grid: Iterable[float],
    cfg: McConfig = McConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    mod = str(cfg.modulation).strip().lower()
    if mod not in ("qpsk", "16qam"):
        raise ValueError("modulation must be 'qpsk' or '16qam'")

    bits_per_sym = 2 if mod == "qpsk" else 4
    n_bits = int(max(cfg.n_bits_per_block, bits_per_sym))
    n_bits = int((n_bits // bits_per_sym) * bits_per_sym)
    n_syms = n_bits // bits_per_sym
    n_blocks = int(max(cfg.n_blocks, 1))

    rng = np.random.default_rng(int(cfg.seed))

    sinr_db_arr = np.asarray(list(sinr_db_grid), dtype=float).reshape((-1,))
    bler_arr = np.zeros_like(sinr_db_arr, dtype=float)

    for i, snr_db in enumerate(sinr_db_arr):
        snr_lin = float(10.0 ** (float(snr_db) / 10.0))
        # Vectorize over blocks
        bits = rng.integers(0, 2, size=(n_blocks, n_syms, bits_per_sym), dtype=np.int8)
        if mod == "qpsk":
            s = _qpsk_map(bits.reshape((-1, bits_per_sym))).reshape((n_blocks, n_syms))
            y = _awgn(s, snr_lin, rng)
            bhat = _qpsk_demod_hard(y.reshape((-1,))).reshape((n_blocks, n_syms, bits_per_sym))
        else:
            s = _qam16_map(bits.reshape((-1, bits_per_sym))).reshape((n_blocks, n_syms))
            y = _awgn(s, snr_lin, rng)
            bhat = _qam16_demod_hard(y.reshape((-1,))).reshape((n_blocks, n_syms, bits_per_sym))
        bit_err_per_block = np.sum(bits != bhat, axis=(1, 2))
        bler_arr[i] = float(np.mean(bit_err_per_block > 0))

    return sinr_db_arr, np.clip(bler_arr, 0.0, 1.0)


def save_bler_csv(path: str, sinr_db: np.ndarray, bler: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sinr_db", "bler"])
        for x, y in zip(sinr_db.tolist(), bler.tolist()):
            w.writerow([float(x), float(y)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="BLER_A.csv", help="output CSV path")
    ap.add_argument("--mod", default="qpsk", choices=["qpsk", "16qam"])
    ap.add_argument("--bits", type=int, default=1024, help="bits per block")
    ap.add_argument("--blocks", type=int, default=400, help="number of blocks per SINR point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sinr_db", default="-5,-2,0,2,4,6,8,10,12,15", help="comma-separated SINR(dB) points")
    args = ap.parse_args()

    grid = [float(x.strip()) for x in str(args.sinr_db).split(",") if x.strip()]
    cfg = McConfig(modulation=args.mod, n_bits_per_block=args.bits, n_blocks=args.blocks, seed=args.seed)
    sinr_db, bler = simulate_bler_curve(grid, cfg=cfg)
    save_bler_csv(str(args.out), sinr_db, bler)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
