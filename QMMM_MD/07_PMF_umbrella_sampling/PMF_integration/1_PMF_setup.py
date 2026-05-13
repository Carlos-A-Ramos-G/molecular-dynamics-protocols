#!/usr/bin/env python3
"""Generate meta.dat and wham.sh from pmf_config.yaml."""

import argparse
import sys
from pathlib import Path


def _cast(value: str):
    """Convert a YAML scalar string to int, float, or str."""
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
    """Parse a flat key: value YAML file without external dependencies."""
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()   # strip inline comments
            if not line or ":" not in line:
                continue
            key, _, raw = line.partition(":")
            cfg[key.strip()] = _cast(raw.strip())
    required = ["rc_start", "windows", "rc_step", "force",
                "wham_bins", "wham_tol", "temperature", "wham_numpad",
                "meta_file", "pmf_output", "wham_log"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Missing keys in config: {', '.join(missing)}")
    return cfg


def window_centers(rc_start: float, windows: int, rc_step: float) -> list[float]:
    # Use integer arithmetic to avoid float accumulation errors
    return [round(rc_start + i * rc_step, 10) for i in range(windows)]


def write_meta(cfg: dict, out_dir: Path) -> None:
    centers = window_centers(cfg["rc_start"], cfg["windows"], cfg["rc_step"])
    force = cfg["force"]
    meta_path = out_dir / cfg["meta_file"]
    lines = [f"{i + 1}.dat {c:.1f} {force}\n" for i, c in enumerate(centers)]
    meta_path.write_text("".join(lines))
    print(f"Written: {meta_path}  ({len(lines)} windows)")


def write_wham(cfg: dict, out_dir: Path) -> None:
    centers = window_centers(cfg["rc_start"], cfg["windows"], cfg["rc_step"])
    hist_min = centers[0]
    hist_max = centers[-1]
    cmd = (
        f"wham  {hist_min:.1f} {hist_max:.1f}"
        f" {cfg['wham_bins']}"
        f" {cfg['wham_tol']}"
        f" {cfg['temperature']}"
        f" {cfg['wham_numpad']}"
        f" {cfg['meta_file']}"
        f" {cfg['pmf_output']}"
        f" > {cfg['wham_log']}\n"
    )
    wham_path = out_dir / "wham.sh"
    wham_path.write_text(f"#!/bin/bash\n\n{cmd}")
    wham_path.chmod(0o755)
    print(f"Written: {wham_path}")
    print(f"  wham range: [{hist_min:.1f}, {hist_max:.1f}]  bins={cfg['wham_bins']}  T={cfg['temperature']} K")


def main():
    parser = argparse.ArgumentParser(description="Generate meta.dat and wham.sh for PMF integration.")
    parser.add_argument("--config", default="pmf_config.yaml",
                        help="Path to YAML config file (default: pmf_config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    out_dir = config_path.parent

    write_meta(cfg, out_dir)
    write_wham(cfg, out_dir)


if __name__ == "__main__":
    main()
