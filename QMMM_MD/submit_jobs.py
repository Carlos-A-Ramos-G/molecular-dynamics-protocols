#!/usr/bin/env python3
"""
Submit SLURM jobs for QM/MM umbrella sampling with correct dependencies.

Dependency chain:
    equil → scan[1] → scan[2] → … → scan[N]
                ↓         ↓               ↓
            pmf[1]    pmf[2]          pmf[N]

Each scan window depends on the previous one (sequential restarts).
Each PMF window depends only on its own scan window (independent productions).

Usage:
    python submit_jobs.py [--config config.yaml] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found.  Install with:  pip install pyyaml  or  conda install pyyaml")


# ── Constants ─────────────────────────────────────────────────────────────────

STAGES = {
    "equil": "05_QMMM_restraint_free_simulations",
    "scan":  "06_scan_umbrella_sampling",
    "pmf":   "07_PMF_umbrella_sampling",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def check_inputs(root: Path, windows: int) -> None:
    """Abort early if any expected job script is missing."""
    equil_dir = root / STAGES["equil"]
    scan_dir  = root / STAGES["scan"]
    pmf_dir   = root / STAGES["pmf"]

    expected = (
        [equil_dir / "run_1_eq.cmd"]
        + [scan_dir / f"{w}.cmd" for w in range(1, windows + 1)]
        + [pmf_dir  / f"{w}.cmd" for w in range(1, windows + 1)]
    )
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        sys.exit(
            "ERROR: missing job scripts — run generate_inputs.py first:\n  "
            + "\n  ".join(missing)
        )


def sbatch(script: Path, chdir: Path, dependency: str = None, dry_run: bool = False) -> str:
    """Run sbatch and return the job ID string.  Prints the command in dry-run mode."""
    cmd = ["sbatch", "--parsable", f"--chdir={chdir}"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script))

    if dry_run:
        print("  " + " ".join(cmd))
        return "<dry-run-id>"

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit QM/MM umbrella sampling SLURM jobs."
    )
    parser.add_argument(
        "--config", default="config.yaml", type=Path,
        help="path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print sbatch commands without submitting anything",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"ERROR: config file not found: {config_path}")

    cfg = load_config(config_path)
    root = config_path.parent
    windows = int(cfg["windows"])

    equil_dir = root / STAGES["equil"]
    scan_dir  = root / STAGES["scan"]
    pmf_dir   = root / STAGES["pmf"]

    check_inputs(root, windows)

    if args.dry_run:
        print("DRY RUN — commands that would be submitted:\n")

    equil_id = sbatch(equil_dir / "run_1_eq.cmd", equil_dir, dry_run=args.dry_run)
    print(f"  equilibration  job_id={equil_id}")

    prev_id = equil_id
    for window in range(1, windows + 1):
        scan_id = sbatch(
            scan_dir / f"{window}.cmd", scan_dir,
            dependency=prev_id, dry_run=args.dry_run,
        )
        sbatch(
            pmf_dir / f"{window}.cmd", pmf_dir,
            dependency=scan_id, dry_run=args.dry_run,
        )
        print(f"  window {window:3d}      scan_id={scan_id}")
        prev_id = scan_id

    if not args.dry_run:
        print("\nAll jobs submitted.")


if __name__ == "__main__":
    main()
