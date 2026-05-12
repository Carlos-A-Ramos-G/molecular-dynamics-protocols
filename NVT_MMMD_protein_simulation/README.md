# Molecular Mechanics Molecular Dynamics  (MM MD) preparation -> NVT production (stages 00 -> 04)

This is the first half of a QM/MM string-method PMF workflow.
However, it can also be used as a standalone protocol for running MM MD production simulations.
`run_upto_NVT.py` configures and runs stages 00 (preparation) through 04
(NVT production) for an enzyme + inhibitor system.

Two execution modes are supported:

- **cluster** — generates `run_gpu` and submits it with `sbatch` (SLURM).
- **local** — generates `run_local` and runs it directly with `bash` (local GPU workstation, no scheduler needed).

Stages 05 -> 08 (QM/MM equilibration, scan, second QM/MM equilibration,
adaptive string method) are launched separately and will get their own driver.

## What you need

- AMBER (>= 18, tested with 24):
  - **cluster**: must be a `module` on your cluster (`amber_module` in config).
  - **local**: either on PATH, or set `amber_home` to your AMBER install directory
    (the driver will `source $amber_home/amber.sh`).
- Python >= 3.8 (standard library only for the driver).
- `propka` (optional, if you want to predict protonation states):
      pip install propka
- A PDB file plus any non-standard residue parameter files
  (`*.lib`, `*.frcmod`) placed in `00_prep/`.

## Quick start — cluster (SLURM)

1. Drop your PDB and parameter files into `00_prep/`.
2. Edit `00_prep/leap_structure_tmp` to reference your non-standard
   residue libraries (the `loadpdb` line is rewritten by the driver).
3. Copy `config.ini.example` -> `config.ini` and set `execution_mode = cluster`,
   then fill in `account`, `amber_module`, and walltimes.
4. Render and submit:

   ```
   python run_upto_NVT.py -c config.ini --submit
   ```

   Or run interactively:

   ```
   python run_upto_NVT.py
   ```

## Quick start — local GPU workstation

1. Drop your PDB and parameter files into `00_prep/`.
2. Edit `00_prep/leap_structure_tmp` as above.
3. Copy `config.ini.example` -> `config.ini` and set `execution_mode = local`.
   If AMBER is not on your PATH, also set `amber_home = /path/to/amber24`.
4. Render the run script (and optionally launch immediately):

   ```
   python run_upto_NVT.py -c config.ini          # generates run_local
   python run_upto_NVT.py -c config.ini --submit  # generates + launches
   ```

   Or run interactively:

   ```
   python run_upto_NVT.py
   ```

   To launch manually afterwards:

   ```
   bash run_local
   ```

## What the driver writes

| File | Role |
|------|------|
| `00_prep/leap_structure`          | tleap input with the user's PDB filled in |
| `02_heat/heat_GPU.in`             | dt / nstlim / target temperature patched in place |
| `04_NVT/prod1.in`                 | dt / nstlim / temperature patched in place |
| `04_NVT/run_template`             | `__TOPOLOGY__` substituted — cluster NVT worker |
| `04_NVT/run_local_template`       | `__TOPOLOGY__` substituted — local NVT worker |
| `run_gpu`  *(cluster mode)*       | Master SLURM submit script (00 -> 04), from `run_gpu_template` |
| `run_local` *(local mode)*        | Master bash script (00 -> 04), from `run_local_template` |

## How HMR works in the pipeline

If `use_hmr = yes`:

- `00_prep` runs `cpptraj -i HMR.ccptraj` after `tleap`, producing
  `structure_HMR.parm7`.
- All `pmemd.cuda -p` references switch to `structure_HMR.parm7`.
- All MD timesteps switch to `dt = 0.004`; nstlim values are halved
  to keep the wall time per stage constant.

If `use_hmr = no`, the original 2 fs timestep with `structure.parm7` is used.

## Single-day vs multi-day cluster

The 04_NVT scheme is the same in either case; only the launcher arguments
change. They are exposed as `chunks_per_job` and `walltime_nvt` in
`config.ini`:

| Cluster type | `chunks_per_job` | `walltime_nvt` |
|--------------|------------------|----------------|
| 5-day queue  | `total_chunks` (e.g. 5) | `5-00:00:00` |
| 1-day queue  | 1                       | `1-00:00:00` |

For a 1-day queue the worker self-resubmits the next chunk via `sbatch`
when its window is finished, so the user only ever submits the first job.

## Local mode — NVT production

In local mode, `04_NVT/script.sh` is called with the `--local` flag.
It instantiates `04_NVT/run_local_template` into `run_NVT_1_local.sh`
and runs all chunks sequentially via `bash` (no sbatch, no scheduler).
If `chunks_per_job < total_chunks`, each worker script chains to the next
one by calling `bash run_NVT_<n>_local.sh` directly.

Typical local invocation (runs all 5 × 100 ns chunks back-to-back):

```
# in config.ini:
execution_mode  = local
chunks_per_job  = 5      ; run all 5 chunks in one go
total_chunks    = 5
```
