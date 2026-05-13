#!/usr/bin/env python3
"""Plot PMF_error_estimate: mean free energy with ± standard deviation band."""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")


def read_pmf_error(path: Path) -> tuple:
    rc, mean, std = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rc.append(float(parts[0]))
            mean.append(float(parts[1]))
            std.append(float(parts[2]))
    return rc, mean, std


def plot(rc, mean, std, out_path: Path) -> None:
    upper = [m + s for m, s in zip(mean, std)]
    lower = [m - s for m, s in zip(mean, std)]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(rc, lower, upper, alpha=0.25, color="steelblue", label="± std dev")
    ax.plot(rc, mean, color="steelblue", linewidth=1.8, label="Mean PMF")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")

    ax.set_xlabel("Reaction coordinate (Å)", fontsize=12)
    ax.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)", fontsize=12)
    ax.set_title("Potential of Mean Force", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PMF with error band.")
    parser.add_argument("--input", default="PMF_error_estimate",
                        help="PMF error estimate file (default: PMF_error_estimate)")
    parser.add_argument("--output", default="PMF_error_estimate.png",
                        help="Output image file (default: PMF_error_estimate.png)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    rc, mean, std = read_pmf_error(in_path)
    plot(rc, mean, std, Path(args.output))


if __name__ == "__main__":
    main()
