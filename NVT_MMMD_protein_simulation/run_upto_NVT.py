#!/usr/bin/env python3
"""
run_upto_NVT.py
===============

Interactive driver for stages 00 -> 04 of the QM/MM PMF workflow:
    00_prep  -> 01_min -> 02_heat -> 03_equil -> 04_NVT (production)

Supports two execution modes:

  * cluster  -- generates a SLURM submit script (run_gpu) launched with sbatch.
  * local    -- generates a plain bash script (run_local) for a local GPU
                workstation; no scheduler required.

What it does
------------
1.  Asks the user (interactively) or reads an INI config:
      * input PDB filename (must already live in 00_prep/)
      * whether to run propka to assign protonation states
      * whether to use hydrogen-mass repartitioning (HMR -> 4 fs)
      * production temperature (default 300 K)
      * production length: ns_per_chunk x total_chunks (default 5 x 100 ns)
      * execution_mode: cluster or local
      * cluster: SLURM walltimes / module / partition / account
      * local:   amber_home (optional; leave blank if pmemd.cuda is on PATH)
2.  Validates that 00_prep/ contains the PDB and any non-standard residue
    parameter files referenced from leap_structure_tmp.
3.  Optionally runs propka on the PDB and pauses so you can update
    protonation states (HID/HIE/HIP, ASH/ASP, GLH/GLU, LYN/LYS) in the
    PDB before proceeding.
4.  Patches stage-specific input files in place:
      * 02_heat/heat_GPU.in    (dt, nstlim, ramp end temperature)
      * 04_NVT/prod1.in        (dt, nstlim, tempi, temp0)
      * 00_prep/leap_structure (loadpdb line)
    The 03_equil generator reads its parameters from environment
    variables that run_gpu / run_local sets, so no in-place edit is needed.
5.  Renders the master run script from the appropriate template:
      * cluster: run_gpu_template   -> run_gpu   (submit with: sbatch run_gpu)
      * local:   run_local_template -> run_local  (run with:   bash run_local)
6.  Patches __TOPOLOGY__ in the matching NVT worker template:
      * cluster: 04_NVT/run_template
      * local:   04_NVT/run_local_template
7.  Optionally submits (sbatch) or runs (bash) the master script.

Usage
-----
    python run_upto_NVT.py                 # interactive
    python run_upto_NVT.py -c config.ini   # non-interactive
    python run_upto_NVT.py --dry-run       # render files; skip propka & launch
    python run_upto_NVT.py -c config.ini --submit
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPL_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    # workflow
    "pdb":            "MUT_INH_monA.pdb",
    "use_propka":     "no",
    "propka_path":    "propka3",
    "use_hmr":        "yes",
    "temperature":    "300.0",
    "ns_per_chunk":   "100",
    "total_chunks":   "5",
    # execution mode
    "execution_mode": "cluster",   # "cluster" (SLURM) or "local" (GPU workstation)
    "amber_home":     "",          # local only: AMBER install dir; sources amber.sh
                                   #   leave blank if pmemd.cuda is already on PATH
    # slurm / cluster
    "chunks_per_job": "5",
    "walltime_master":"1-00:00:00",
    "walltime_nvt":   "5-00:00:00",
    "amber_module":   "apps/amber/24",
    "account":        "CHEM031804",
    "job_name":       "INH_MUT",
    # numerics
    "min_cycles_cap":       "100",
    "convergence_threshold":"1.0e-3",
    # 03_equil chunk lengths in nanoseconds (per cycle)
    "equil_npt_ns":   "1.25",
    "equil_nvt_ns":   "5.0",
    # 02_heat duration in ps
    "heat_ps":        "200",
    # behaviour
    "submit":         "no",
}

YES = {"y", "yes", "true", "1", "on"}


def yn(s: str) -> bool:
    return str(s).strip().lower() in YES


# ---------------------------------------------------------------------------
# Config loading + interactive prompts
# ---------------------------------------------------------------------------
def ask(prompt: str, default: str, choices=None) -> str:
    suffix = ""
    if choices:
        suffix += f" ({'/'.join(choices)})"
    if default:
        suffix += f" [{default}]"
    while True:
        try:
            ans = input(f"{prompt}{suffix}: ").strip() or default
        except EOFError:
            return default
        if choices and ans.lower() not in [c.lower() for c in choices]:
            print(f"  -> please answer one of: {', '.join(choices)}")
            continue
        return ans


def load_or_prompt(args) -> dict:
    cfg = dict(DEFAULTS)
    if args.config:
        parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        parser.read(args.config)
        if "run_upto_NVT" not in parser:
            sys.exit(f"ERROR: {args.config} has no [run_upto_NVT] section")
        cfg.update({k: str(v) for k, v in parser["run_upto_NVT"].items()})
        if args.submit:
            cfg["submit"] = "yes"
        return cfg

    print("\nrun_upto_NVT.py -- interactive setup (Enter to accept defaults)\n")
    cfg["pdb"] = ask("PDB filename inside 00_prep/", cfg["pdb"])
    cfg["use_propka"] = ask("Run propka to assign protonation states?",
                            cfg["use_propka"], choices=["yes", "no"])
    if yn(cfg["use_propka"]):
        cfg["propka_path"] = ask("propka executable (or 'propka3' on PATH)",
                                 cfg["propka_path"])
    cfg["use_hmr"] = ask("Use hydrogen-mass repartitioning (4 fs timestep)?",
                         cfg["use_hmr"], choices=["yes", "no"])
    cfg["temperature"] = ask("Production temperature (K)", cfg["temperature"])
    cfg["ns_per_chunk"] = ask("ns per production chunk", cfg["ns_per_chunk"])
    cfg["total_chunks"] = ask("Total chunks (sampling = total_chunks * ns_per_chunk)",
                              cfg["total_chunks"])

    cfg["execution_mode"] = ask(
        "Execution mode", cfg["execution_mode"],
        choices=["cluster", "local"])

    if cfg["execution_mode"] == "cluster":
        cfg["chunks_per_job"] = ask(
            "Chunks per SLURM job (1 for short queues, =total_chunks for long queues)",
            cfg["chunks_per_job"])
        cfg["walltime_master"] = ask("Walltime for master job (00->03)",
                                     cfg["walltime_master"])
        cfg["walltime_nvt"] = ask("Walltime per NVT chunk job", cfg["walltime_nvt"])
        cfg["amber_module"] = ask("AMBER module string", cfg["amber_module"])
        cfg["account"]  = ask("SLURM account", cfg["account"])
        cfg["job_name"] = ask("SLURM job name", cfg["job_name"])
        cfg["submit"]   = ask("sbatch run_gpu when done?", cfg["submit"],
                              choices=["yes", "no"])
    else:
        cfg["amber_home"] = ask(
            "AMBER install directory (blank = pmemd.cuda already on PATH)",
            cfg.get("amber_home", ""))
        cfg["job_name"] = ask("Run name (for labelling only)", cfg["job_name"])
        cfg["chunks_per_job"] = ask(
            "Chunks per run_local invocation (usually = total_chunks)",
            cfg["total_chunks"])
        cfg["submit"] = ask("Launch run_local when done?", cfg["submit"],
                            choices=["yes", "no"])
    return cfg


# ---------------------------------------------------------------------------
# Validation + propka
# ---------------------------------------------------------------------------
def validate_inputs(cfg: dict) -> None:
    pdb = REPL_ROOT / "00_prep" / cfg["pdb"]
    if not pdb.exists():
        sys.exit(f"ERROR: {pdb} not found.\n"
                 f"Place the PDB and any non-standard residue parameter files "
                 f"(e.g. INH.lib, INH_ff14SB.frcmod) in {pdb.parent}/ first.")
    leap_tmp = REPL_ROOT / "00_prep" / "leap_structure_tmp"
    if not leap_tmp.exists():
        sys.exit(f"ERROR: {leap_tmp} missing. Cannot generate leap input.")
    if yn(cfg["use_hmr"]):
        hmr = REPL_ROOT / "00_prep" / "HMR.ccptraj"
        if not hmr.exists():
            sys.exit(f"ERROR: HMR requested but {hmr} not found.")
    # Local mode: warn early if pmemd.cuda cannot be found.
    mode = cfg.get("execution_mode", "cluster").strip().lower()
    if mode == "local":
        amber_home = cfg.get("amber_home", "").strip()
        if not amber_home and not shutil.which("pmemd.cuda"):
            print("WARNING: pmemd.cuda not found on PATH and amber_home is not set.\n"
                  "  Set amber_home in config.ini (path to your AMBER install dir)\n"
                  "  or ensure pmemd.cuda is on PATH before running run_local.")


def run_propka(cfg: dict) -> None:
    if not yn(cfg["use_propka"]):
        return
    pdb = REPL_ROOT / "00_prep" / cfg["pdb"]
    print(f"\nRunning propka on {pdb} ...")
    try:
        subprocess.check_call([cfg["propka_path"], pdb.name], cwd=str(pdb.parent))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        sys.exit(f"ERROR: propka failed: {e}\n"
                 "Install with `pip install propka` or set propka_path.")
    pka = pdb.with_suffix(".pka")
    print(f"propka output: {pka}\n"
          "Inspect the .pka file and edit the PDB to set residue names\n"
          "(HID/HIE/HIP, ASH/ASP, GLH/GLU, LYN/LYS) per the predicted pKa.\n")
    input("Press Enter once the PDB is updated (Ctrl-C to abort): ")


# ---------------------------------------------------------------------------
# Render leap input
# ---------------------------------------------------------------------------
def render_leap(cfg: dict) -> None:
    src = REPL_ROOT / "00_prep" / "leap_structure_tmp"
    dst = REPL_ROOT / "00_prep" / "leap_structure"
    text = src.read_text()
    text = re.sub(r"loadpdb\s+\S+", f"loadpdb {cfg['pdb']}", text)
    dst.write_text(text)
    print(f"  wrote {dst.relative_to(REPL_ROOT)}")


# ---------------------------------------------------------------------------
# Patch heat / prod inputs in place
# ---------------------------------------------------------------------------
def _sub(text: str, key: str, new: str) -> str:
    """Replace a key=value pair (AMBER mdin style) in `text`."""
    return re.sub(rf"({key}\s*=\s*)[\d.eE+-]+", rf"\g<1>{new}", text, count=1)


def patch_heat(cfg: dict, dt: float) -> None:
    path = REPL_ROOT / "02_heat" / "heat_GPU.in"
    text = path.read_text()
    heat_steps = int(round(float(cfg["heat_ps"]) / dt))
    ramp_end   = int(round(0.8 * heat_steps))   # ramp finishes ~80% of the run
    text = _sub(text, "nstlim", str(heat_steps))
    text = _sub(text, "dt", f"{dt}")
    text = _sub(text, "value2", str(cfg["temperature"]))
    text = re.sub(r"(istep2\s*=\s*)\d+", rf"\g<1>{ramp_end}", text, count=1)
    path.write_text(text)
    print(f"  patched {path.relative_to(REPL_ROOT)} "
          f"(dt={dt}, nstlim={heat_steps}, T->{cfg['temperature']})")


def patch_prod(cfg: dict, dt: float) -> None:
    path = REPL_ROOT / "04_NVT" / "prod1.in"
    text = path.read_text()
    nstlim = int(round(float(cfg["ns_per_chunk"]) * 1000.0 / dt))
    text = _sub(text, "nstlim", str(nstlim))
    text = _sub(text, "dt", f"{dt}")
    text = _sub(text, "tempi", str(cfg["temperature"]))
    text = _sub(text, "temp0", str(cfg["temperature"]))
    path.write_text(text)
    print(f"  patched {path.relative_to(REPL_ROOT)} "
          f"(dt={dt}, nstlim={nstlim}, T->{cfg['temperature']})")


# ---------------------------------------------------------------------------
# Render master run script + NVT worker template
# ---------------------------------------------------------------------------
def _amber_setup_block(cfg: dict) -> str:
    """Return the shell snippet that makes pmemd.cuda available locally."""
    amber_home = cfg.get("amber_home", "").strip()
    if amber_home:
        return (f"# Source AMBER environment\n"
                f'source "{amber_home}/amber.sh"')
    return ("# pmemd.cuda is assumed to be on PATH\n"
            "# (set amber_home in config.ini to source a specific AMBER install)")


def render_master(cfg: dict, dt: float, topology: str) -> None:
    mode = cfg.get("execution_mode", "cluster").strip().lower()

    if yn(cfg["use_hmr"]):
        hmr_block = ("# Hydrogen-mass repartitioning -> 4 fs timestep\n"
                     "cpptraj -i HMR.ccptraj")
    else:
        hmr_block = "# HMR disabled"

    npt_steps = int(round(float(cfg["equil_npt_ns"]) * 1000.0 / dt))
    nvt_steps = int(round(float(cfg["equil_nvt_ns"]) * 1000.0 / dt))

    # Substitutions common to both modes.
    common_subs = {
        "TOPOLOGY":        topology,
        "HMR_BLOCK":       hmr_block,
        "MIN_CYCLES_CAP":  cfg["min_cycles_cap"],
        "CONV_THRESH":     cfg["convergence_threshold"],
        "TEMPERATURE":     cfg["temperature"],
        "DT":              f"{dt}",
        "EQUIL_NPT_STEPS": str(npt_steps),
        "EQUIL_NVT_STEPS": str(nvt_steps),
        "CHUNKS_PER_JOB":  cfg["chunks_per_job"],
        "TOTAL_CHUNKS":    cfg["total_chunks"],
        "JOB_NAME":        cfg["job_name"],
    }

    if mode == "local":
        tmpl_path     = REPL_ROOT / "run_local_template"
        out_path      = REPL_ROOT / "run_local"
        nvt_tmpl_path = REPL_ROOT / "04_NVT" / "run_local_template"
        subs = {**common_subs,
                "AMBER_SETUP": _amber_setup_block(cfg)}
    else:
        tmpl_path     = REPL_ROOT / "run_gpu_template"
        out_path      = REPL_ROOT / "run_gpu"
        nvt_tmpl_path = REPL_ROOT / "04_NVT" / "run_template"
        subs = {**common_subs,
                "WALLTIME_MASTER": cfg["walltime_master"],
                "WALLTIME_NVT":    cfg["walltime_nvt"],
                "AMBER_MODULE":    cfg["amber_module"],
                "ACCOUNT":         cfg["account"]}

    tmpl = tmpl_path.read_text()
    for k, v in subs.items():
        tmpl = tmpl.replace(f"__{k}__", str(v))

    out_path.write_text(tmpl)
    out_path.chmod(0o755)
    print(f"  wrote {out_path.relative_to(REPL_ROOT)}")

    # Patch __TOPOLOGY__ in the NVT worker template.
    nvt_tmpl = nvt_tmpl_path.read_text()
    nvt_tmpl = nvt_tmpl.replace("__TOPOLOGY__", topology)
    nvt_tmpl_path.write_text(nvt_tmpl)
    print(f"  patched {nvt_tmpl_path.relative_to(REPL_ROOT)} "
          f"(__TOPOLOGY__ -> {topology})")


# ---------------------------------------------------------------------------
# Optional launch / submit
# ---------------------------------------------------------------------------
def maybe_submit(cfg: dict) -> None:
    mode = cfg.get("execution_mode", "cluster").strip().lower()

    if mode == "local":
        if not yn(cfg["submit"]):
            print("\nDone. To run:  bash run_local")
            return
        print("\nLaunching run_local ...")
        subprocess.check_call(["bash", "run_local"], cwd=str(REPL_ROOT))
        return

    # cluster mode
    if not yn(cfg["submit"]):
        print("\nDone. To submit:  sbatch run_gpu")
        return
    if not shutil.which("sbatch"):
        print("\nNo `sbatch` on PATH; skipping submission.")
        print("To submit later:  sbatch run_gpu")
        return
    print("\nSubmitting run_gpu ...")
    subprocess.check_call(["sbatch", "run_gpu"], cwd=str(REPL_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", help="INI config file (skip prompts)")
    p.add_argument("--dry-run", action="store_true",
                   help="Render files; skip propka and launch")
    p.add_argument("--submit", action="store_true",
                   help="Force launch (sbatch or bash) even if config says no")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_or_prompt(args)

    print("\nValidating inputs ...")
    validate_inputs(cfg)

    if not args.dry_run:
        run_propka(cfg)

    use_hmr  = yn(cfg["use_hmr"])
    dt       = 0.004 if use_hmr else 0.002
    topology = "structure_HMR.parm7" if use_hmr else "structure.parm7"

    print("\nRendering input files ...")
    render_leap(cfg)
    patch_heat(cfg, dt)
    patch_prod(cfg, dt)
    render_master(cfg, dt, topology)

    if not args.dry_run:
        maybe_submit(cfg)
    else:
        print("\n--dry-run set; skipping launch.")


if __name__ == "__main__":
    main()
