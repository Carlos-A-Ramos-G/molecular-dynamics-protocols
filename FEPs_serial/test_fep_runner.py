"""
Tests for fep_runner.py — local mode.

Runs `setup --mode local` in a temp directory and validates the generated
AMBER TI input files against expected values from config.yaml.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).parent
N_WINDOWS = 9

# Expected michaelis alchemical masks (from config.yaml FCE_to_ACE example).
MICHAELIS_MASKS = {
    "timask1": "':608-611'",
    "timask2": "':612-615'",
    "scmask1": "'@9291-9294'",
    "scmask2": "'@9358-9361'",
}

# 9-point Gauss-Legendre nodes mapped to [0, 1], sorted ascending.
_nodes, _ = np.polynomial.legendre.leggauss(N_WINDOWS)
EXPECTED_LAMBDAS = sorted((_nodes + 1) / 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_param(text: str, key: str) -> str:
    """Return the value of *key* from an AMBER &cntrl block.

    Handles both quoted masks (e.g. ':608-611') and bare numbers.
    """
    # Quoted value — captures everything between the single quotes so that
    # commas inside the mask (e.g. '@9358,9360,9361') are not truncated.
    m = re.search(rf"{key}\s*=\s*'([^']*)'", text)
    if m:
        return f"'{m.group(1)}'"
    # Bare numeric value
    m = re.search(rf"{key}\s*=\s*([0-9.]+)", text)
    if m:
        return m.group(1)
    return None


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
# GL lambda values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
def test_michaelis_lambda_matches_gl_quadrature(generated, window):
    """clambda in each michaelis window must equal the 9-point GL node."""
    ti = generated / "FCE_to_ACE" / "michaelis" / "replica_1" / str(window) / f"ti_{window}.in"
    assert ti.exists(), f"Missing: {ti}"
    val = float(_extract_param(ti.read_text(), "clambda"))
    assert abs(val - EXPECTED_LAMBDAS[window - 1]) < 1e-4, (
        f"Window {window}: clambda={val:.5f}, expected={EXPECTED_LAMBDAS[window - 1]:.5f}"
    )


# ---------------------------------------------------------------------------
# Masks — michaelis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
@pytest.mark.parametrize("mask_key", ["timask1", "timask2", "scmask1", "scmask2"])
def test_michaelis_masks(generated, window, mask_key):
    """Alchemical masks must match the expected values from config.yaml."""
    ti = (
        generated / "FCE_to_ACE" / "michaelis" / "replica_1" / str(window) / f"ti_{window}.in"
    )
    assert ti.exists(), f"Missing: {ti}"
    val = _extract_param(ti.read_text(), mask_key)
    assert val == MICHAELIS_MASKS[mask_key], (
        f"Window {window} [{mask_key}]: got {val!r}, expected {MICHAELIS_MASKS[mask_key]!r}"
    )


# ---------------------------------------------------------------------------
# Simulation parameters — michaelis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", range(1, N_WINDOWS + 1))
@pytest.mark.parametrize("param,expected", [
    ("nstlim", "5000000"),
    ("dt",     "0.001"),
    ("ntwe",   "1000"),
    ("ntwx",   "10000"),
])
def test_michaelis_sim_params(generated, window, param, expected):
    """Simulation parameters must match config.yaml production settings."""
    ti = generated / "FCE_to_ACE" / "michaelis" / "replica_1" / str(window) / f"ti_{window}.in"
    assert ti.exists()
    val = _extract_param(ti.read_text(), param)
    assert val == expected, f"Window {window} [{param}]: got {val!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# run_local.sh generated for both systems
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("system", ["michaelis", "ligand"])
def test_run_local_sh_exists(generated, system):
    """run_local.sh must be generated for every system in local mode."""
    script = generated / "FCE_to_ACE" / system / "run_local.sh"
    assert script.exists(), f"run_local.sh missing for {system}"
    assert script.stat().st_mode & 0o111, f"run_local.sh not executable for {system}"


@pytest.mark.parametrize("system", ["michaelis", "ligand"])
def test_run_local_sh_contains_all_windows(generated, system):
    """run_local.sh must reference every lambda window."""
    script = generated / "FCE_to_ACE" / system / "run_local.sh"
    content = script.read_text()
    for w in range(1, N_WINDOWS + 1):
        assert f"ti_{w}.in" in content, (
            f"run_local.sh for {system} does not reference ti_{w}.in"
        )


# ---------------------------------------------------------------------------
# Ligand masks — sanity check (must NOT be overwritten by michaelis values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask_key,expected", [
    ("timask1", "':1-4'"),
    ("timask2", "':5-8'"),
    ("scmask1", "'@1-4'"),
    ("scmask2", "'@68-71'"),
])
def test_ligand_masks_correct(generated, mask_key, expected):
    """Ligand arm masks must be the solvated-ligand residue/atom ranges."""
    ti = generated / "FCE_to_ACE" / "ligand" / "replica_1" / "1" / "ti_1.in"
    assert ti.exists()
    val = _extract_param(ti.read_text(), mask_key)
    assert val == expected, f"Ligand [{mask_key}]: got {val!r}, expected {expected!r}"
