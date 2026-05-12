#!/usr/bin/env python3
"""
Generate AMBER input files and SLURM job scripts for QM/MM umbrella sampling.

Usage:
    python generate_inputs.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found.  Install with:  pip install pyyaml  or  conda install pyyaml")


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_KEYS = [
    "parm", "geom", "scheme",
    "qlevel", "eecut", "qmmask", "qcharge",
    "atom1", "atom2", "atom3",
    "coor0", "windows", "scan_step",
    "force_equil", "force_scan", "force_pmf",
]

STAGES = {
    "equil": "05_QMMM_restraint_free_simulations",
    "scan":  "06_scan_umbrella_sampling",
    "pmf":   "07_PMF_umbrella_sampling",
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"ERROR: missing config keys: {', '.join(missing)}")
    return cfg


# ── Template filling ──────────────────────────────────────────────────────────

def fill_template(template: Path, output: Path, subs: dict) -> None:
    """Replace PLACEHOLDER tokens in template and write to output."""
    text = template.read_text()
    for placeholder, value in subs.items():
        text = text.replace(placeholder, str(value))
    output.write_text(text)


def qm_subs(cfg: dict, **extra) -> dict:
    """Base substitutions for all QM/MM input and job-script templates."""
    subs = {
        "QMMASK":  f"'{cfg['qmmask']}'",    # AMBER masks require surrounding single quotes
        "QCHARGE": str(cfg["qcharge"]),
        "QLEVEL":  cfg["qlevel"],
        "EECUT":   str(cfg["eecut"]),
        "SCHEME":  cfg["scheme"],
        "PARM":    cfg["parm"],
        "GEOM":    cfg["geom"],
    }
    subs.update(extra)
    return subs


def restr_subs(cfg: dict, force: int, coor: float, vali: float, valf: float) -> dict:
    """Substitutions for NMR flat-bottom restraint files."""
    return {
        "ATOM1": str(cfg["atom1"]),
        "ATOM2": str(cfg["atom2"]),
        "ATOM3": str(cfg["atom3"]),
        "FORCE": str(force),
        "COOR":  f"{coor:.1f}",
        "VALI":  f"{vali:.1f}",
        "VALF":  f"{valf:.1f}",
    }


# ── Stage builders ────────────────────────────────────────────────────────────

def setup_equil(root: Path, templates: Path, cfg: dict) -> None:
    """Generate input files for the QM/MM restraint-free equilibration (stage 05)."""
    stage_dir = root / STAGES["equil"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"Setting up {stage_dir.name}")

    fill_template(
        templates / "run_1_eq_template",
        stage_dir / "run_1_eq.cmd",
        qm_subs(cfg),
    )

    coor0 = float(cfg["coor0"])
    restr_tpl = templates / cfg.get("restr_template", "restr_template")
    fill_template(
        restr_tpl,
        stage_dir / "restr",
        restr_subs(cfg, cfg["force_equil"], coor0, coor0 - 1.0, coor0 + 1.0),
    )

    fill_template(
        templates / "in_1_eq_free_template",
        stage_dir / "in",
        qm_subs(cfg),
    )


def setup_windows(root: Path, templates: Path, cfg: dict) -> None:
    """Generate per-window input and job files for the scan (06) and PMF (07) stages."""
    coor0     = float(cfg["coor0"])
    step_size = float(cfg["scan_step"])
    restr_tpl = templates / cfg.get("restr_template", "restr_template")

    for stage_key, force in [("scan", cfg["force_scan"]), ("pmf", cfg["force_pmf"])]:
        stage_dir = root / STAGES[stage_key]
        stage_dir.mkdir(parents=True, exist_ok=True)

        for window in range(1, int(cfg["windows"]) + 1):
            val = round(coor0 + (window - 1) * step_size, 1)
            print(f"  {STAGES[stage_key]}  window {window:3d}  coord={val:.1f}")

            fill_template(
                restr_tpl,
                stage_dir / f"{window}.restr",
                restr_subs(cfg, force, val, val - 10.0, val + 10.0),
            )

            if stage_key == "scan":
                fill_template(
                    templates / "in_2_restrained_template",
                    stage_dir / f"{window}.in",
                    qm_subs(cfg, STEP=str(window)),
                )
                fill_template(
                    templates / "run_2_scan_template",
                    stage_dir / f"{window}.cmd",
                    qm_subs(cfg, STEP=str(window), PREV=str(window - 1)),
                )

            else:  # pmf
                for label in ("eq", "PMF"):
                    fill_template(
                        templates / f"in_3_{label}_template",
                        stage_dir / f"{label}_{window}.in",
                        qm_subs(cfg, STEP=str(window)),
                    )
                fill_template(
                    templates / "run_3_PMF_template",
                    stage_dir / f"{window}.cmd",
                    qm_subs(cfg, STEP=str(window)),
                )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate QM/MM umbrella sampling input files from templates."
    )
    parser.add_argument(
        "--config", default="config.yaml", type=Path,
        help="path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        sys.exit(f"ERROR: config file not found: {config_path}")

    cfg = load_config(config_path)
    root = config_path.parent
    templates = root / "template_files"

    if not templates.is_dir():
        sys.exit(f"ERROR: template directory not found: {templates}")

    for key in ("parm", "geom"):
        p = root / cfg[key]
        if not p.exists():
            print(f"WARNING: {key} file not found: {p}", file=sys.stderr)

    setup_equil(root, templates, cfg)
    setup_windows(root, templates, cfg)
    print("\nDone. Review generated files, then submit with:  python submit_jobs.py")


if __name__ == "__main__":
    main()
