"""
Tests for fep_runner.py — local mode.

Runs `setup --mode local` in a temp directory and validates the generated
AMBER TI input files against expected values drawn directly from config.yaml.
Changing config.yaml for a new mutation automatically updates all expectations.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load config — all expected values are derived from it
# ---------------------------------------------------------------------------

_cfg = yaml.safe_load((REPO_DIR / "config.yaml").read_text())

SYSTEM_NAME     = f"{_cfg['resnew']}_to_{_cfg['resold']}"
N_WINDOWS       = _cfg["n_lambdas"]
BOUNDED_MASKS   = {k: _cfg["systems"]["bounded"][k]
                   for k in ("timask1", "timask2", "scmask1", "scmask2")}
UNBOUNDED_MASKS = {k: _cfg["systems"]["unbounded"][k]
                   for k in ("timask1", "timask2", "scmask1", "scmask2")}
_prod           = _cfg["simulation"]["prod"]

# Gauss-Legendre nodes mapped to [0, 1], sorted ascending.
_nodes, _ = np.polynomial.legendre.leggauss(N_WINDOWS)
EXPECTED_LAMBDAS = sorted((_nodes + 1) / 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_param(text: str, key: str) -> str:
    """Return the value of *key* from an AMBER &cntrl block.

    Handles both quoted masks (e.g. ':608-611') and bare numbers.
    """
    m = re.search(rf"{key}\s*=\s*'([^']*)'", text)
    if m:
        return f"'{m.group(1)}'"
    m = re.search(rf"{key}\s*=\s*([0-9.]+)", text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Fixture — generate files once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Run setup --mode local once; return the tmp directory."""
    tmp = tmp_path_factory.mktemp("fep_local")
    shutil.copy(REPO_DIR / "config.yaml", tmp / "config.yaml")
    shutil.copy(REPO_DIR / "fep_runner.py", tmp / "fep_runner.py")

    result = subprocess.run(
        [sys.executable, "fep_runner.py", "setup", "--mode", "local"],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"fep_runner.py setup --mode local failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return tmp


# ---------------------------------------------------------------------------
# GL lambda values — bounded leg
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
def test_bounded_lambda_matches_gl_quadrature(generated, window):
    """clambda in each bounded window must equal the GL node."""
    ti = generated / SYSTEM_NAME / "bounded" / "replica_1" / str(window) / f"ti_{window}.in"
    assert ti.exists(), f"Missing: {ti}"
    val = float(_extract_param(ti.read_text(), "clambda"))
    assert abs(val - EXPECTED_LAMBDAS[window - 1]) < 1e-4, (
        f"Window {window}: clambda={val:.5f}, expected={EXPECTED_LAMBDAS[window - 1]:.5f}"
    )


# ---------------------------------------------------------------------------
# Masks — bounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
@pytest.mark.parametrize("mask_key", ["timask1", "timask2", "scmask1", "scmask2"])
def test_bounded_masks(generated, window, mask_key):
    """Alchemical masks must match config.yaml bounded values."""
    ti = generated / SYSTEM_NAME / "bounded" / "replica_1" / str(window) / f"ti_{window}.in"
    assert ti.exists(), f"Missing: {ti}"
    val = _extract_param(ti.read_text(), mask_key)
    assert val == BOUNDED_MASKS[mask_key], (
        f"Window {window} [{mask_key}]: got {val!r}, expected {BOUNDED_MASKS[mask_key]!r}"
    )


# ---------------------------------------------------------------------------
# Simulation parameters — bounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
@pytest.mark.parametrize("param,expected", [
    ("nstlim", str(_prod["nstlim"])),
    ("dt",     str(_prod["dt"])),
    ("ntwe",   str(_prod["ntwe"])),
    ("ntwx",   str(_prod["ntwx"])),
])
def test_bounded_sim_params(generated, window, param, expected):
    """Simulation parameters must match config.yaml production settings."""
    ti = generated / SYSTEM_NAME / "bounded" / "replica_1" / str(window) / f"ti_{window}.in"
    assert ti.exists()
    val = _extract_param(ti.read_text(), param)
    assert val == expected, f"Window {window} [{param}]: got {val!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# run_local.sh generated for both legs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_run_local_sh_exists(generated, system):
    """run_local.sh must be generated for every leg in local mode."""
    script = generated / SYSTEM_NAME / system / "run_local.sh"
    assert script.exists(), f"run_local.sh missing for {system}"
    assert script.stat().st_mode & 0o111, f"run_local.sh not executable for {system}"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_run_local_sh_contains_all_windows(generated, system):
    """run_local.sh must reference every lambda window."""
    script = generated / SYSTEM_NAME / system / "run_local.sh"
    content = script.read_text()
    for w in range(1, N_WINDOWS + 1):
        assert f"ti_{w}.in" in content, (
            f"run_local.sh for {system} does not reference ti_{w}.in"
        )


# ---------------------------------------------------------------------------
# Masks — unbounded (must NOT be overwritten by bounded values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask_key,expected", [
    (k, UNBOUNDED_MASKS[k]) for k in ("timask1", "timask2", "scmask1", "scmask2")
])
def test_unbounded_masks_correct(generated, mask_key, expected):
    """Unbounded arm masks must match config.yaml unbounded values."""
    ti = generated / SYSTEM_NAME / "unbounded" / "replica_1" / "1" / "ti_1.in"
    assert ti.exists()
    val = _extract_param(ti.read_text(), mask_key)
    assert val == expected, f"Unbounded [{mask_key}]: got {val!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Fixture — serial mode (SLURM scripts, no submission)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_serial(tmp_path_factory):
    """Run setup --mode serial once; return the tmp directory."""
    tmp = tmp_path_factory.mktemp("fep_serial")
    shutil.copy(REPO_DIR / "config.yaml", tmp / "config.yaml")
    shutil.copy(REPO_DIR / "fep_runner.py", tmp / "fep_runner.py")
    result = subprocess.run(
        [sys.executable, "fep_runner.py", "setup", "--mode", "serial"],
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"fep_runner.py setup --mode serial failed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return tmp


# ---------------------------------------------------------------------------
# Generic SLURM header — all slurm.gpu keys must appear in EQUILIBRATION.cmd
# ---------------------------------------------------------------------------

_gpu_resources = _cfg["slurm"]["gpu"]


@pytest.mark.parametrize("key,value", _gpu_resources.items())
def test_slurm_header_gpu_keys(generated_serial, key, value):
    """Every key in slurm.gpu must produce a #SBATCH directive in EQUILIBRATION.cmd."""
    cmd = (generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd").read_text()
    assert f"#SBATCH --{key}={value}" in cmd, (
        f"Expected '#SBATCH --{key}={value}' in EQUILIBRATION.cmd"
    )


# ---------------------------------------------------------------------------
# EQUILIBRATION.cmd — combined min + heating + equil script
# ---------------------------------------------------------------------------

_gpu_exec = _cfg["execution_command"]["gpu"]
_cpu_exec = _cfg["execution_command"]["cpu"].format(
    ntasks=_cfg["slurm"]["cpu"]["ntasks"]
)

_mid = (N_WINDOWS + 1) // 2
_sample_windows = sorted({1, _mid, N_WINDOWS})


def test_slurm_equilibration_cmd_exists(generated_serial):
    """EQUILIBRATION.cmd must be generated (replaces the old FEP_MIN/FEP_EQUIL pair)."""
    cmd = generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd"
    assert cmd.exists(), "EQUILIBRATION.cmd missing"
    assert cmd.stat().st_mode & 0o111, "EQUILIBRATION.cmd not executable"


def test_slurm_equilibration_contains_gpu_command(generated_serial):
    """EQUILIBRATION.cmd must use the GPU execution command for min and heating."""
    cmd = (generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd").read_text()
    assert _gpu_exec in cmd, f"Expected '{_gpu_exec}' in EQUILIBRATION.cmd"


def test_slurm_equilibration_contains_cpu_command(generated_serial):
    """EQUILIBRATION.cmd must use the CPU execution command for equilibration."""
    cmd = (generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd").read_text()
    assert _cpu_exec in cmd, f"Expected '{_cpu_exec}' in EQUILIBRATION.cmd"


def test_slurm_equilibration_contains_all_three_stages(generated_serial):
    """EQUILIBRATION.cmd must reference min.in, heating.in, and equil.in."""
    cmd = (generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd").read_text()
    for stage in ("min.in", "heating.in", "equil.in"):
        assert stage in cmd, f"EQUILIBRATION.cmd missing reference to {stage}"


def test_slurm_equilibration_chains_to_prod(generated_serial):
    """EQUILIBRATION.cmd must submit the central production window."""
    cmd = (generated_serial / SYSTEM_NAME / "unbounded" / "EQUILIBRATION.cmd").read_text()
    assert f"FEP_PROD_{_mid}.cmd" in cmd


# ---------------------------------------------------------------------------
# execution_command — GPU command in prod scripts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", _sample_windows)
def test_slurm_prod_uses_gpu_execution_command(generated_serial, window):
    """FEP_PROD_N.cmd must contain the gpu execution command from config."""
    cmd = (
        generated_serial / SYSTEM_NAME / "unbounded"
        / "replica_1" / str(window) / f"FEP_PROD_{window}.cmd"
    ).read_text()
    assert _gpu_exec in cmd, f"Expected '{_gpu_exec}' in FEP_PROD_{window}.cmd"


# ---------------------------------------------------------------------------
# heating.in — generated for both legs in local mode
# ---------------------------------------------------------------------------

_heat_cfg = _cfg["simulation"]["heating"]


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_heating_in_exists(generated, system):
    """heating.in must be generated for every leg."""
    heating = generated / SYSTEM_NAME / system / "heating.in"
    assert heating.exists(), f"heating.in missing for {system}"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
@pytest.mark.parametrize("param,expected", [
    ("nstlim", str(_heat_cfg["nstlim"])),
    ("dt",     str(_heat_cfg["dt"])),
    ("nmropt", "1"),
    ("ntb",    "1"),
    ("ntp",    "0"),
])
def test_heating_in_params(generated, system, param, expected):
    """heating.in must contain correct NVT parameters."""
    content = (generated / SYSTEM_NAME / system / "heating.in").read_text()
    val = _extract_param(content, param)
    assert val == expected, f"heating.in [{system}][{param}]: got {val!r}, expected {expected!r}"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_heating_in_contains_wt_block(generated, system):
    """heating.in must contain the temperature ramp &wt block."""
    content = (generated / SYSTEM_NAME / system / "heating.in").read_text()
    assert "&wt" in content, f"heating.in for {system} missing &wt temperature ramp"
    assert "TEMP0" in content


# ---------------------------------------------------------------------------
# run_local.sh — heating step and GPU equil
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_run_local_sh_contains_heating(generated, system):
    """run_local.sh must include the NVT heating step."""
    script = (generated / SYSTEM_NAME / system / "run_local.sh").read_text()
    assert "heating.in" in script, f"run_local.sh for {system} missing heating step"
    assert "heating.rst7" in script, f"run_local.sh for {system} missing heating.rst7 output"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_run_local_sh_no_cpu_amber(generated, system):
    """run_local.sh must not reference CPU_AMBER — equilibration now runs on GPU."""
    script = (generated / SYSTEM_NAME / system / "run_local.sh").read_text()
    assert "CPU_AMBER" not in script, f"run_local.sh for {system} still references CPU_AMBER"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_run_local_equil_reads_from_heating(generated, system):
    """Equilibration step in run_local.sh must use heating.rst7 as input coordinates."""
    script = (generated / SYSTEM_NAME / system / "run_local.sh").read_text()
    assert "-c heating.rst7" in script, (
        f"run_local.sh for {system}: equil step must read from heating.rst7"
    )


# ---------------------------------------------------------------------------
# Extended minimisation — maxcyc and ntpr from config
# ---------------------------------------------------------------------------

_min_cfg = _cfg["simulation"]["min"]


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_min_maxcyc(generated, system):
    """min.in maxcyc must match config."""
    content = (generated / SYSTEM_NAME / system / "min.in").read_text()
    val = _extract_param(content, "maxcyc")
    assert val == str(_min_cfg["maxcyc"]), f"min.in [{system}] maxcyc: got {val!r}"


@pytest.mark.parametrize("system", ["bounded", "unbounded"])
def test_min_ntpr(generated, system):
    """min.in ntpr must match config."""
    content = (generated / SYSTEM_NAME / system / "min.in").read_text()
    val = _extract_param(content, "ntpr")
    assert val == str(_min_cfg["ntpr"]), f"min.in [{system}] ntpr: got {val!r}"
