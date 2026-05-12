#!/usr/bin/env python3
"""
FEP Gaussian Quadrature workflow manager for AMBER TI calculations.

Commands
--------
  python fep_runner.py setup [--mode MODE] [--submit]
  python fep_runner.py analyse [--tail N]

Modes
-----
  serial   (default) SLURM cluster; one job runs at a time.  Replicas are
           chained end-to-end: all windows of replica 1 finish before replica 2
           begins.  Chain order per replica: equil → mid → mid-1 → … → 1 →
           mid+1 → … → N → replica+1/mid.

  parallel           SLURM cluster with no GPU limit.  After equilibration all
           replicas are submitted simultaneously.  Within each replica the
           central window fans out bidirectionally so that both neighbours start
           as soon as the central window finishes:
               1 ← 2 ← 3 ← 4 ← 5 → 6 → 7 → 8 → 9

  local              No SLURM.  Generates run_local.sh for a single-GPU
           workstation.  Everything runs sequentially inside one script; no job
           scheduler is needed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")


# =============================================================================
# Gauss-Legendre quadrature
# =============================================================================

def compute_gl_quadrature(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes (lambda values) and weights mapped to [0, 1].

    numpy.polynomial.legendre.leggauss returns nodes on [-1, 1] with weights
    that integrate to 2.  The map x → (x+1)/2 rescales to [0, 1] and halves
    the weights so they still sum to 1.
    """
    nodes_11, weights_11 = np.polynomial.legendre.leggauss(n)
    return (nodes_11 + 1) / 2, weights_11 / 2


def _middle(n_windows: int) -> int:
    """1-based index of the central lambda window."""
    return (n_windows + 1) // 2


# =============================================================================
# AMBER input templates
# =============================================================================

_MIN_TEMPLATE = """\
minimisation
 &cntrl
   imin = 1, ntmin = 2, maxcyc = {maxcyc},
   ntpr = 500, ntwe = 20,
   dx0 = 1.0D-7,
   ntb = 1, ntxo = 1,

   icfe = 1, ifsc = 1, clambda = 0.5, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /

 &ewald
 /
"""

_EQUIL_TEMPLATE = """\
TI equilibration
 &cntrl
   imin = 0,
   nstlim = {nstlim}, irest = 0, ntx = 1, dt = {dt},
   ntt = 3, temp0 = 300.0, gamma_ln = 2.0, ig = -1,
   ntc = 1, ntf = 1,
   ntb = 2, ntp = 1, pres0 = 1.01325, taup = 2.0, barostat = 2,
   ntwe = {ntwx}, ntwx = {ntwx}, ntpr = {ntwx}, ntwr = {ntwx},

   icfe = 1, ifsc = 1, clambda = 0.5, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /

 &ewald
 /
"""

_PROD_TEMPLATE = """\
TI simulation
 &cntrl
   imin = 0, nstlim = {nstlim}, irest = 0, ntx = 1, dt = {dt},
   ntt = 3, temp0 = 298.0, gamma_ln = 2.0, ig = -1,
   ntc = 1, ntf = 1,
   ntb = 2, ntp = 1, pres0 = 1.01325, taup = 2.0, barostat = 2,
   ntwe = {ntwe}, ntwx = {ntwx}, ntpr = {ntwe}, ntwr = {ntwe},

   icfe = 1, ifsc = 1, clambda = {clambda:.5f}, scalpha = 0.5, scbeta = 12.0,
   logdvdl = 0,
   timask1 = {timask1}, timask2 = {timask2},
   scmask1 = {scmask1}, scmask2 = {scmask2},
 /

 &ewald
 /
"""


# =============================================================================
# SLURM script helpers
# =============================================================================

def _sbatch_header(job_name: str, resources: dict) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=mpi_%j.out",
        "#SBATCH --error=mpi_%j.err",
        f"#SBATCH --ntasks={resources['ntasks']}",
    ]
    for key in ("gres", "partition", "qos"):
        if key in resources:
            lines.append(f"#SBATCH --{key}={resources[key]}")
    return "\n".join(lines)


def _module_block(module: str) -> str:
    return f"\nmodule purge\nmodule load {module}\n"


def _equil_cpptraj_params(cfg: dict, n_replicas: int) -> tuple[int, int, int]:
    """Return (start_frame, end_frame, step) for the cpptraj equil extraction."""
    equil = cfg["simulation"]["equil"]
    total_frames = equil["nstlim"] // equil["ntwx"]
    start_frame = total_frames // 2
    if n_replicas == 1:
        return total_frames, total_frames, 1
    step = (total_frames - start_frame) // (n_replicas - 1)
    return start_frame, total_frames, step


# ---------------------------------------------------------------------------
# Minimisation SLURM script (identical for all cluster modes)
# ---------------------------------------------------------------------------

def _gen_min_cmd(resnew: str, sys_label: str, cfg: dict) -> str:
    header = _sbatch_header(f"{resnew}-min-{sys_label}", cfg["slurm"]["gpu"])
    mods = _module_block(cfg["amber"]["cuda_module"])
    return "\n".join([
        header, mods,
        "srun $AMBERHOME/bin/pmemd.cuda -i min.in -c ti.inpcrd -p ti.prmtop -O \\",
        "    -o min.out -inf min.info -e min.en -r min.rst -l min.log",
        "",
        "sbatch FEP_EQUIL.cmd",
        "",
    ])


# ---------------------------------------------------------------------------
# Equilibration SLURM script  (differs between serial and parallel)
# ---------------------------------------------------------------------------

def _gen_equil_cmd(resnew: str, sys_label: str, n_replicas: int,
                   middle: int, cfg: dict, mode: str) -> str:
    resources = cfg["slurm"]["cpu"]
    n_tasks = resources["ntasks"]
    header = _sbatch_header(f"{resnew}-equil-{sys_label}", resources)
    mods = _module_block(cfg["amber"]["cpu_module"])
    start, end, step = _equil_cpptraj_params(cfg, n_replicas)

    lines = [
        header, mods,
        f"srun -n {n_tasks} $AMBERHOME/bin/pmemd.MPI -O \\",
        "    -i equil.in -c min.rst -p ti.prmtop \\",
        "    -o equil.out -inf equil.info -e equil.en \\",
        "    -r equil.rst -x equil.nc -l equil.log",
        "",
        f"# Extract {n_replicas} restart file(s) from the second half of the equilibration",
        f"# trajectory.  cpptraj appends .1 … .{n_replicas} to the output name because",
        f"# multiple frames are written; each replica's central window uses its own",
        f"# numbered file as independent starting coordinates.",
        "cpptraj <<_EOF",
        "parm ti.prmtop",
        f"trajin equil.nc {start} {end} {step}",
        "trajout equil.rst7",
        "_EOF",
        "",
    ]

    if mode == "parallel":
        # Submit all replicas' central window simultaneously
        lines += [
            "top=$(pwd)",
            f"for r in $(seq 1 {n_replicas}); do",
            f'    (cd "$top/replica_${{r}}/{middle}" && sbatch FEP_PROD_{middle}.cmd)',
            "done",
            "",
        ]
    else:  # serial
        lines += [
            "top=$(pwd)",
            f"cd $top/replica_1/{middle}",
            f"sbatch FEP_PROD_{middle}.cmd",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Production SLURM scripts  (differ between serial and parallel)
# ---------------------------------------------------------------------------

def _prod_submissions(window: int, replica: int, n_windows: int,
                      n_replicas: int, mode: str) -> list[tuple[str, str]]:
    """
    Return a list of (directory, script) pairs to sbatch after this window.

    serial  — one submission; replicas chain end-to-end via the last window.
    parallel — the central window fans out to both neighbours simultaneously;
               all other windows submit only their single downstream neighbour;
               endpoints (1 and N) submit nothing; no cross-replica chaining.
    """
    mid = _middle(n_windows)

    if mode == "parallel":
        if window == mid:
            return [
                (f"../{mid - 1}", f"FEP_PROD_{mid - 1}.cmd"),
                (f"../{mid + 1}", f"FEP_PROD_{mid + 1}.cmd"),
            ]
        if 1 < window < mid:
            return [(f"../{window - 1}", f"FEP_PROD_{window - 1}.cmd")]
        if window == 1 or window == n_windows:
            return []
        if mid < window < n_windows:
            return [(f"../{window + 1}", f"FEP_PROD_{window + 1}.cmd")]
        return []

    # serial
    if window == mid:
        return [(f"../{mid - 1}", f"FEP_PROD_{mid - 1}.cmd")]
    if 1 < window < mid:
        return [(f"../{window - 1}", f"FEP_PROD_{window - 1}.cmd")]
    if window == 1:
        return [(f"../{mid + 1}", f"FEP_PROD_{mid + 1}.cmd")]
    if mid < window < n_windows:
        return [(f"../{window + 1}", f"FEP_PROD_{window + 1}.cmd")]
    # window == n_windows
    if replica < n_replicas:
        return [(f"../../replica_{replica + 1}/{mid}", f"FEP_PROD_{mid}.cmd")]
    return []


def _gen_prod_cmd(window: int, replica: int, n_windows: int, n_replicas: int,
                  resnew: str, sys_label: str, lambdas: np.ndarray,
                  cfg: dict, mode: str) -> str:
    """
    Generate the SLURM production script for one window / replica pair.

    Coordinate seeding is the same in all cluster modes:
      • central window  — reads equil.rst7.{replica}
      • window < mid    — reads from the window above (FUTUR = window+1)
      • window > mid    — reads from the window below (PAST  = window-1)

    The only mode-dependent part is which job(s) to submit afterwards.
    """
    mid = _middle(n_windows)
    prod = cfg["simulation"]["prod"]
    last_frame = prod["nstlim"] // prod["ntwx"]

    # Starting coordinates
    if window == mid:
        coords = f"../../equil.rst7.{replica}"
    elif window < mid:
        coords = f"../{window + 1}/ti001_{window + 1}_final.rst7"
    else:
        coords = f"../{window - 1}/ti001_{window - 1}_final.rst7"

    # Windows 1 and N are chain endpoints; no downstream window reads from them.
    save_rst = (window != 1) and (window != n_windows)

    submissions = _prod_submissions(window, replica, n_windows, n_replicas, mode)

    job_name = f"{resnew}_R{replica}.{window}-{sys_label}"
    header = _sbatch_header(job_name, cfg["slurm"]["gpu"])
    mods = _module_block(cfg["amber"]["cuda_module"])

    lines = [
        header, mods,
        f"srun $AMBERHOME/bin/pmemd.cuda -i ti_{window}.in -c {coords} -p ti.prmtop -O \\",
        f"    -o ti001_{window}.out -inf ti001_{window}.info -e ti001_{window}.en \\",
        f"    -r ti001_{window}.rst -x ti001_{window}.nc -l ti001_{window}.log",
        "",
    ]

    if save_rst:
        lines += [
            "cpptraj <<_EOF",
            "parm ti.prmtop",
            f"trajin ti001_{window}.nc {last_frame} {last_frame} 1",
            f"trajout ti001_{window}_final.rst7",
            "run",
            "_EOF",
            "",
        ]

    if len(submissions) > 1:
        # Bidirectional fan-out: use subshells so both sbatch calls happen
        # from their respective window directories without losing the CWD.
        for d, c in submissions:
            lines.append(f"(cd {d} && sbatch {c})")
        lines.append("")
    elif submissions:
        d, c = submissions[0]
        lines += [f"cd {d}", f"sbatch {c}", ""]

    return "\n".join(lines)


# =============================================================================
# Local (no-SLURM) script
# =============================================================================

def _gen_local_script(sys_label: str, n_windows: int, n_replicas: int,
                      lambdas: np.ndarray, cfg: dict) -> str:
    """
    Generate run_local.sh: a single sequential bash script for a one-GPU
    workstation.  All stages — min, equil, and every production window for
    every replica — run one after another in the same process.

    Window order within each replica:
        mid, mid-1, …, 1, mid+1, …, N
    (same as the serial cluster mode, just without sbatch).

    Usage:
        bash run_local.sh                        # foreground
        nohup bash run_local.sh > run.log 2>&1 & # background
    """
    mid = _middle(n_windows)
    prod = cfg["simulation"]["prod"]
    last_frame = prod["nstlim"] // prod["ntwx"]

    start, end, step = _equil_cpptraj_params(cfg, n_replicas)

    # Descending from mid to 1, then ascending from mid+1 to N
    window_order = list(range(mid, 0, -1)) + list(range(mid + 1, n_windows + 1))

    L = [
        "#!/bin/bash",
        "# Sequential FEP run for a single GPU — no SLURM required.",
        "#",
        "# Usage:",
        "#   bash run_local.sh                         # foreground",
        "#   nohup bash run_local.sh > run.log 2>&1 &  # background",
        "#",
        "# Requires: AMBERHOME set, pmemd.cuda and cpptraj in $AMBERHOME/bin.",
        "",
        "set -euo pipefail",
        "",
        'SYSDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SYSDIR"',
        "",
        ': "${AMBERHOME:?Please set AMBERHOME before running this script}"',
        'AMBER="$AMBERHOME/bin/pmemd.cuda"',
        'CPPTRAJ="$AMBERHOME/bin/cpptraj"',
        "",
        'log() { echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"; }',
        "",
        "# ---- Minimisation ------------------------------------------------",
        'log "Minimisation"',
        '$AMBER -i min.in -c ti.inpcrd -p ti.prmtop -O \\',
        '    -o min.out -inf min.info -e min.en -r min.rst -l min.log',
        "",
        "# ---- Equilibration (CPU → GPU here for single-workstation use) ---",
        'log "Equilibration"',
        '$AMBER -i equil.in -c min.rst -p ti.prmtop -O \\',
        '    -o equil.out -inf equil.info -e equil.en \\',
        '    -r equil.rst -x equil.nc -l equil.log',
        "",
        f"# Extract {n_replicas} restart file(s) from the second half of the equil",
        "# trajectory.  cpptraj appends .1 … .N to the output filename.",
        '$CPPTRAJ <<_EOF',
        "parm ti.prmtop",
        f"trajin equil.nc {start} {end} {step}",
        "trajout equil.rst7",
        "_EOF",
        "",
    ]

    for replica in range(1, n_replicas + 1):
        L += [
            f"# ---- Replica {replica} " + "-" * max(1, 50 - len(str(replica))),
            f'log "Replica {replica}/{n_replicas}"',
            "",
        ]
        for window in window_order:
            clambda = lambdas[window - 1]

            if window == mid:
                coords = f"../../equil.rst7.{replica}"
            elif window < mid:
                coords = f"../{window + 1}/ti001_{window + 1}_final.rst7"
            else:
                coords = f"../{window - 1}/ti001_{window - 1}_final.rst7"

            save_rst = (window != 1) and (window != n_windows)

            L += [
                f'log "  Window {window}/{n_windows}  lambda={clambda:.5f}"',
                f'cd "$SYSDIR/replica_{replica}/{window}"',
                f'$AMBER -i ti_{window}.in -c {coords} -p ti.prmtop -O \\',
                f'    -o ti001_{window}.out -inf ti001_{window}.info \\',
                f'    -e ti001_{window}.en   -r ti001_{window}.rst \\',
                f'    -x ti001_{window}.nc   -l ti001_{window}.log',
            ]

            if save_rst:
                L += [
                    '$CPPTRAJ <<_EOF',
                    "parm ti.prmtop",
                    f"trajin ti001_{window}.nc {last_frame} {last_frame} 1",
                    f"trajout ti001_{window}_final.rst7",
                    "run",
                    "_EOF",
                ]

            L.append("")

    L += ['log "All simulations complete."', ""]
    return "\n".join(L)


# =============================================================================
# Setup command
# =============================================================================

def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def setup(cfg: dict, submit: bool = False, mode: str = "serial") -> None:
    system_name = cfg["system_name"]
    n_lambdas = cfg["n_lambdas"]
    n_replicas = cfg["replicates"]

    if n_lambdas < 3:
        sys.exit("n_lambdas must be >= 3.")
    if mode not in ("serial", "parallel", "local"):
        sys.exit(f"Unknown mode '{mode}'. Choose: serial, parallel, local.")

    lambdas, weights = compute_gl_quadrature(n_lambdas)
    mid = _middle(n_lambdas)

    print(f"System  : {system_name}")
    print(f"Mode    : {mode}")
    print(f"Windows : {n_lambdas}  (central window: {mid})")
    print(f"Replicas: {n_replicas}")
    print(f"\n{'Win':>4}  {'λ':>9}  {'weight':>9}")
    print(f"{'---':>4}  {'---------':>9}  {'---------':>9}")
    for i, (lam, w) in enumerate(zip(lambdas, weights), 1):
        print(f"{i:>4}  {lam:>9.5f}  {w:>9.5f}")
    print(f"{'':>4}  {'sum':>9}  {weights.sum():>9.5f}")
    print()

    base = Path(system_name)
    sim = cfg["simulation"]

    for sys_label, sys_cfg in cfg["systems"].items():
        sys_dir = base / sys_label
        sys_dir.mkdir(parents=True, exist_ok=True)

        # Symlinks to parameter files at system level
        setup_dir = Path("../../setup_files") / system_name
        _symlink(setup_dir / sys_cfg["parameters"], sys_dir / "ti.prmtop")
        _symlink(setup_dir / sys_cfg["coordinates"], sys_dir / "ti.inpcrd")

        mask = dict(
            timask1=sys_cfg["timask1"], timask2=sys_cfg["timask2"],
            scmask1=sys_cfg["scmask1"], scmask2=sys_cfg["scmask2"],
        )

        # AMBER input files at system level (λ=0.5 for min and equil)
        (sys_dir / "min.in").write_text(
            _MIN_TEMPLATE.format(maxcyc=sim["min"]["maxcyc"], **mask)
        )
        (sys_dir / "equil.in").write_text(
            _EQUIL_TEMPLATE.format(
                nstlim=sim["equil"]["nstlim"],
                dt=sim["equil"]["dt"],
                ntwx=sim["equil"]["ntwx"],
                **mask,
            )
        )

        # Per-replica / per-window: AMBER production inputs + topology symlink
        for replica in range(1, n_replicas + 1):
            for w_idx, clambda in enumerate(lambdas, 1):
                win_dir = sys_dir / f"replica_{replica}" / str(w_idx)
                win_dir.mkdir(parents=True, exist_ok=True)
                _symlink(
                    Path("../../../../setup_files") / system_name / sys_cfg["parameters"],
                    win_dir / "ti.prmtop",
                )
                (win_dir / f"ti_{w_idx}.in").write_text(
                    _PROD_TEMPLATE.format(
                        nstlim=sim["prod"]["nstlim"],
                        dt=sim["prod"]["dt"],
                        ntwe=sim["prod"]["ntwe"],
                        ntwx=sim["prod"]["ntwx"],
                        clambda=clambda,
                        **mask,
                    )
                )

        # Mode-specific job scripts
        resnew = sys_cfg["resnew"]

        if mode == "local":
            _write_exe(
                sys_dir / "run_local.sh",
                _gen_local_script(sys_label, n_lambdas, n_replicas, lambdas, cfg),
            )
        else:
            _write_exe(sys_dir / "FEP_MIN.cmd",
                       _gen_min_cmd(resnew, sys_label, cfg))
            _write_exe(sys_dir / "FEP_EQUIL.cmd",
                       _gen_equil_cmd(resnew, sys_label, n_replicas, mid, cfg, mode))
            for replica in range(1, n_replicas + 1):
                for w_idx, _ in enumerate(lambdas, 1):
                    win_dir = sys_dir / f"replica_{replica}" / str(w_idx)
                    _write_exe(
                        win_dir / f"FEP_PROD_{w_idx}.cmd",
                        _gen_prod_cmd(w_idx, replica, n_lambdas, n_replicas,
                                      resnew, sys_label, lambdas, cfg, mode),
                    )

        print(f"  [{sys_label}] files written → {sys_dir}/")

    print(f"\nSetup complete ({mode} mode).")

    if submit:
        if mode == "local":
            # Build an ordered list of (script, log) pairs from config["systems"].
            # Legs are chained with && so a single GPU is never double-booked:
            # leg 1 must finish before leg 2 starts.
            pairs = [
                (
                    (base / sl / "run_local.sh").resolve(),
                    (base / sl / "run.log").resolve(),
                )
                for sl in cfg["systems"]
            ]
            chain = " && ".join(
                f"bash {script} > {log} 2>&1" for script, log in pairs
            )
            proc = subprocess.Popen(
                chain,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"\nAll legs launched in background (chain PID {proc.pid}).")
            for sl, (_, log) in zip(cfg["systems"], pairs):
                print(f"  [{sl}] log → {log}")
            print(f"\n  Monitor : tail -f <log>")
            print(f"  Stop    : kill {proc.pid}")
        else:
            for sys_label in cfg["systems"]:
                sys_dir = base / sys_label
                result = subprocess.run(
                    ["sbatch", "FEP_MIN.cmd"],
                    capture_output=True, text=True, cwd=sys_dir,
                )
                if result.returncode == 0:
                    print(f"  [{sys_label}] {result.stdout.strip()}")
                else:
                    print(f"  [{sys_label}] sbatch failed: {result.stderr.strip()}",
                          file=sys.stderr)


# =============================================================================
# Analysis command (unchanged)
# =============================================================================

def _extract_dvdl(en_file: Path, tail_lines: int) -> np.ndarray:
    """
    Parse dV/dλ values from an AMBER energy file.

    AMBER writes one 'L9' record per ntwe steps during TI production.
    Field 6 (1-indexed) holds the dV/dλ value.  We keep only the last
    `tail_lines` records to discard the initial equilibration period.
    """
    values: list[float] = []
    with open(en_file) as fh:
        for line in fh:
            if line.startswith(" L9") or line.startswith("L9"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        values.append(float(parts[5]))
                    except ValueError:
                        pass
    if not values:
        raise RuntimeError(f"No L9 records found in {en_file}")
    return np.array(values[-tail_lines:])


def _system_average(sys_label: str, base: Path, n_replicas: int,
                    n_lambdas: int, weights: np.ndarray,
                    tail_lines: int) -> tuple[float, float]:
    replica_dg: list[float] = []
    for replica in range(1, n_replicas + 1):
        window_means = [
            _extract_dvdl(
                base / sys_label / f"replica_{replica}" / str(w)
                / f"ti001_{w}.en",
                tail_lines,
            ).mean()
            for w in range(1, n_lambdas + 1)
        ]
        dg = float(np.dot(window_means, weights))
        print(f"    replica {replica}: ΔG = {dg:9.3f} kcal/mol")
        replica_dg.append(dg)

    mean = float(np.mean(replica_dg))
    std = float(np.std(replica_dg))
    print(f"    mean         : ΔG = {mean:9.3f} ± {std:.3f} kcal/mol")
    return mean, std


def analyse(cfg: dict, tail_lines: int = 4000) -> None:
    system_name = cfg["system_name"]
    n_lambdas = cfg["n_lambdas"]
    n_replicas = cfg["replicates"]
    _, weights = compute_gl_quadrature(n_lambdas)
    base = Path(system_name)

    print(f"\n{'=' * 60}")
    print(f"  Analysis : {system_name}")
    print(f"  Windows  : {n_lambdas}    Replicas : {n_replicas}")
    print(f"  dV/dλ records used per window : last {tail_lines}")
    print(f"{'=' * 60}")

    results: dict[str, tuple[float, float]] = {}
    for sys_label in cfg["systems"]:
        print(f"\n  [{sys_label}]")
        try:
            results[sys_label] = _system_average(
                sys_label, base, n_replicas, n_lambdas, weights, tail_lines
            )
        except (FileNotFoundError, RuntimeError) as exc:
            sys.exit(f"  ERROR: {exc}")

    lig = results["ligand"]
    mic = results["michaelis"]
    ddG = mic[0] - lig[0]
    err = (mic[1] ** 2 + lig[1] ** 2) ** 0.5

    print(f"\n{'=' * 60}")
    print(f"  ΔΔG = ΔG(michaelis) − ΔG(ligand)")
    print(f"  ΔΔG = {ddG:+.3f} ± {err:.3f} kcal/mol")
    print(f"{'=' * 60}\n")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEP Gaussian Quadrature workflow manager for AMBER TI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Configuration file (default: config.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Generate all input files")
    p_setup.add_argument(
        "--mode", default="serial",
        choices=["serial", "parallel", "local"],
        help=(
            "serial   — SLURM, one job at a time, replicas chained (default)\n"
            "parallel — SLURM, all replicas in parallel, bidirectional per replica\n"
            "local    — no SLURM, single run_local.sh for a one-GPU workstation"
        ),
    )
    p_setup.add_argument(
        "--submit", action="store_true",
        help="Submit FEP_MIN.cmd (cluster) or print run instructions (local)",
    )

    p_analyse = sub.add_parser("analyse", help="Extract dV/dλ and compute ΔΔG")
    p_analyse.add_argument(
        "--tail", type=int, default=4000, metavar="N",
        help="Last N dV/dλ records to use per window (default: 4000)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if args.command == "setup":
        setup(cfg, submit=args.submit, mode=args.mode)
    elif args.command == "analyse":
        analyse(cfg, tail_lines=args.tail)


if __name__ == "__main__":
    main()
