#!/usr/bin/env python3
"""Generate one cycle's mdin for stage 03_equil.

Cycles 1..5 are restrained NPT with a backbone restraint that ramps from
15 -> 3 kcal/mol/A^2. Cycle 6 is unrestrained NVT.

Run-wide parameters (temperature, timestep, step counts) are read from
environment variables exported by the master submit script (run_gpu).
Defaults below correspond to a 2 fs timestep without HMR.
"""
import os
import sys

i = int(sys.argv[1])
temp = float(os.environ.get("EQUIL_TEMP", "300.0"))
dt   = float(os.environ.get("EQUIL_DT", "0.002"))
npt_steps = int(os.environ.get("EQUIL_NPT_STEPS", "625000"))   # 1.25 ns @ 2 fs
nvt_steps = int(os.environ.get("EQUIL_NVT_STEPS", "2500000"))  # 5.00 ns @ 2 fs

wt = 15.0 - (i - 1) * 3
print("equilibration")
print(" &cntrl")
print("  imin=0, ")
print("  irest=1,")
print("  ntx=5,")
print("  ntc=2,")
print("  ntf=2,")
print("  ntt=3,")
print(f"  tempi={temp},")
print(f"  temp0={temp},")
print("  ntpr=1000,")
print("  ntwx=1000,")
print(f"  dt={dt},")
if i < 6:
    print(f"  nstlim={npt_steps},")
    print("  ntb=2,")
    print("  ntp=1,")
    print("  ntr = 1,")
    print("  restraintmask='@CA,C,N,O,H&!:WAT',")
    print(f"  restraint_wt={wt},")
else:
    print(f"  nstlim={nvt_steps},")
    print("  ntb=1,")
    print("  ntp=0,")
print("  cut=10,")
print("  gamma_ln=5.0")
print("  iwrap = 1,")
print("!  nmropt = 1,")
print("  ntxo=1,")
print("/")

print("&wt type='END'/")
print("DISANG = restr")
