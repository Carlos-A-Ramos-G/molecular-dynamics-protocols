#!/usr/bin/env python3
"""Estimate PMF statistical error via block averaging.

For each umbrella window, the trajectory data are split into equal-sized
consecutive blocks.  WHAM is run once per block set, producing N PMF
profiles.  Each profile is normalized so that the free energy at the
reactant reference point equals zero.  The mean and standard deviation
across the N profiles are reported as the final PMF with error estimate.

Intermediate files are written to a tmp/ subdirectory.
"""

import argparse
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


# ── Config loader (identical to 1_PMF_setup.py) ───────────────────────────────

def _cast(value: str):
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: Path) -> dict:
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line or ":" not in line:
                continue
            key, _, raw = line.partition(":")
            cfg[key.strip()] = _cast(raw.strip())
    required = ["rc_start", "windows", "rc_step", "force",
                "wham_bins", "wham_tol", "temperature", "wham_numpad",
                "error_chunks", "meta_file", "pmf_output", "wham_log"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Missing keys in config: {', '.join(missing)}")
    return cfg


def window_centers(rc_start: float, windows: int, rc_step: float) -> list:
    return [round(rc_start + i * rc_step, 10) for i in range(windows)]


# ── Data splitting ─────────────────────────────────────────────────────────────

def split_window_files(cfg: dict, work_dir: Path, tmp_dir: Path) -> None:
    n = cfg["error_chunks"]
    for win in range(1, cfg["windows"] + 1):
        src = work_dir / f"../{win}.dat"
        if not src.exists():
            sys.exit(f"Window file not found: {src}")
        lines = [l for l in src.read_text().splitlines() if l.strip()]
        chunk_size = len(lines) // n
        if chunk_size == 0:
            sys.exit(f"{src} has only {len(lines)} lines — cannot split into {n} chunks")
        for chunk in range(n):
            start = chunk * chunk_size
            end = start + chunk_size if chunk < n - 1 else len(lines)
            out = tmp_dir / f"{win}_chunk_{chunk + 1:02d}.dat"
            out.write_text("\n".join(lines[start:end]) + "\n")
    print(f"  Split {cfg['windows']} window files into {n} chunks each.")


# ── Meta files for each chunk ──────────────────────────────────────────────────

def write_chunk_metas(cfg: dict, tmp_dir: Path) -> None:
    centers = window_centers(cfg["rc_start"], cfg["windows"], cfg["rc_step"])
    force = cfg["force"]
    for chunk in range(1, cfg["error_chunks"] + 1):
        lines = []
        for i, rc in enumerate(centers, start=1):
            dat = tmp_dir / f"{i}_chunk_{chunk:02d}.dat"
            lines.append(f"{dat} {rc:.1f} {force}")
        meta = tmp_dir / f"meta_chunk_{chunk:02d}.dat"
        meta.write_text("\n".join(lines) + "\n")


# ── WHAM runs ──────────────────────────────────────────────────────────────────

def run_wham_chunks(cfg: dict, tmp_dir: Path) -> list:
    centers = window_centers(cfg["rc_start"], cfg["windows"], cfg["rc_step"])
    hist_min = f"{centers[0]:.1f}"
    hist_max = f"{centers[-1]:.1f}"
    n = cfg["error_chunks"]
    pmf_paths = []

    for chunk in range(1, n + 1):
        meta    = tmp_dir / f"meta_chunk_{chunk:02d}.dat"
        pmf_out = tmp_dir / f"PMF_chunk_{chunk:02d}"
        log_out = tmp_dir / f"out_chunk_{chunk:02d}"
        cmd = [
            "wham",
            hist_min, hist_max,
            str(cfg["wham_bins"]),
            str(cfg["wham_tol"]),
            str(cfg["temperature"]),
            str(cfg["wham_numpad"]),
            str(meta),
            str(pmf_out),
        ]
        print(f"  wham chunk {chunk:02d}/{n} ...", end="\r", flush=True)
        with open(log_out, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            sys.exit(f"\nwham failed on chunk {chunk} — check {log_out}")
        pmf_paths.append(pmf_out)

    print(f"  {n} wham runs completed.           ")
    return pmf_paths


# ── PMF parsing ────────────────────────────────────────────────────────────────

def parse_pmf(path: Path) -> tuple:
    """Return (rc_list, energy_list) skipping comment lines and non-finite rows."""
    rcs, energies = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rc, energy = float(parts[0]), float(parts[1])
            if energy != float("inf") and energy == energy:  # skip inf / nan
                rcs.append(rc)
                energies.append(energy)
    return rcs, energies


# ── Normalization and averaging ────────────────────────────────────────────────

def find_norm_index(all_energies: list) -> int:
    """Index of the RC bin that is the minimum most often across all profiles."""
    min_indices = [e.index(min(e)) for e in all_energies]
    return Counter(min_indices).most_common(1)[0][0]


def compute_averaged_pmf(pmf_paths: list) -> tuple:
    """Return (rc, mean_dG, std_dG) after block-averaging the normalized PMFs."""
    parsed = [parse_pmf(p) for p in pmf_paths]

    # Trim to the shortest common grid (all grids should match, but be safe)
    min_len = min(len(e) for _, e in parsed)
    rcs         = parsed[0][0][:min_len]
    all_energies = [e[:min_len] for _, e in parsed]

    norm_idx = find_norm_index(all_energies)
    print(f"  Reactant reference: RC = {rcs[norm_idx]:.4f}  (bin index {norm_idx})")

    normalized = [
        [e[i] - e[norm_idx] for i in range(min_len)]
        for e in all_energies
    ]

    cols = list(zip(*normalized))
    means = [statistics.mean(col) for col in cols]
    stds  = [statistics.stdev(col) for col in cols]

    return rcs, means, stds


# ── Output ─────────────────────────────────────────────────────────────────────

def write_output(rcs: list, means: list, stds: list, out_path: Path) -> None:
    header = f"{'#RC':>12}  {'Mean_dG':>14}  {'Std_dG':>14}\n"
    lines = [header]
    for rc, m, s in zip(rcs, means, stds):
        lines.append(f"{rc:>12.4f}  {m:>14.6f}  {s:>14.6f}\n")
    out_path.write_text("".join(lines))
    print(f"Written: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate PMF error by block averaging over trajectory chunks.")
    parser.add_argument("--config", default="pmf_config.yaml",
                        help="Path to config file (default: pmf_config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    cfg     = load_config(config_path)
    work_dir = config_path.parent
    tmp_dir  = work_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"Splitting window data into {cfg['error_chunks']} blocks ...")
    split_window_files(cfg, work_dir, tmp_dir)

    print("Writing chunk meta files ...")
    write_chunk_metas(cfg, tmp_dir)

    print(f"Running {cfg['error_chunks']} WHAM integrations ...")
    pmf_paths = run_wham_chunks(cfg, tmp_dir)

    print("Computing block-averaged PMF ...")
    rcs, means, stds = compute_averaged_pmf(pmf_paths)

    write_output(rcs, means, stds, work_dir / "PMF_error_estimate")


if __name__ == "__main__":
    main()
