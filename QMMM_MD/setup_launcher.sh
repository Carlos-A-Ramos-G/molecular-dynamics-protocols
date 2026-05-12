#!/bin/bash
# QM/MM umbrella sampling setup script
# Generates AMBER input files and SLURM job scripts for:
#   05_QMMM_restraint_free_simulations — initial QM/MM equilibration
#   06_scan_umbrella_sampling          — restrained geometry scan
#   07_PMF_umbrella_sampling           — PMF production windows

set -euo pipefail

# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
# Input structure files (paths relative to this script's directory)
parm="../NVT_MMMD_protein_simulation/00_prep/structure.parm7"
geom="../NVT_MMMD_protein_simulation/04_NVT/structure_NVT_5.rst7"

# Short label used in SLURM job names (e.g. "WT", "D127N")
scheme=""

# QM region
qlevel="AM1"    # theory level: AM1, PM6, DFTB3, …
eecut=12        # non-bonded cutoff for both QM and MM regions (Å)
qmmask="':41,145 &!@C,O,N,H,CA,HA'"
qcharge=0       # total charge of the QM region

# Atoms defining the scanned reaction coordinate (serial numbers)
atom1=2241
atom2=2242
atom3=617

# Scan parameters
coor0=2.9       # starting coordinate value (Å) for window 1
windows=30      # number of umbrella sampling windows
scan_step=0.1   # step size in Å per window (use negative for a reverse scan)
# ─────────────────────────────────────────────────────────────────────────────

restr_template="restr_template"

DIR="$(cd "$(dirname "$0")" && pwd)"
templates="$DIR/template_files"

# ─── VALIDATION ──────────────────────────────────────────────────────────────
for var in parm geom scheme qlevel eecut qmmask qcharge atom1 atom2 atom3 coor0 windows scan_step; do
    if [ -z "${!var}" ]; then
        echo "ERROR: '\$$var' is not set" >&2; exit 1
    fi
done
for f in "$parm" "$geom"; do
    [ -f "$f" ] || echo "WARNING: file not found: $f" >&2
done
[ -d "$templates" ] || { echo "ERROR: template directory not found: $templates" >&2; exit 1; }

mkdir -p "$DIR/05_QMMM_restraint_free_simulations" \
         "$DIR/06_scan_umbrella_sampling" \
         "$DIR/07_PMF_umbrella_sampling"
# ─────────────────────────────────────────────────────────────────────────────

# Escape characters that break sed substitution ( & / \ )
qmmask_escaped=$(printf '%s\n' "$qmmask" | sed 's/[&/\]/\\&/g')

apply_qm_sel() {
    local template="$1" output="$2"
    shift 2
    sed \
        -e "s/QMMASK/$qmmask_escaped/g" \
        -e "s/QCHARGE/$qcharge/g" \
        -e "s/QLEVEL/$qlevel/g" \
        -e "s/EECUT/$eecut/g" \
        "$@" \
        "$template" > "$output"
}

apply_restr_sel() {
    local template="$1" output="$2" force="$3"
    shift 3
    sed \
        -e "s/ATOM1/$atom1/g" \
        -e "s/ATOM2/$atom2/g" \
        -e "s/ATOM3/$atom3/g" \
        -e "s/FORCE/$force/g" \
        "$@" \
        "$template" > "$output"
}

# ─── STAGE 05: QM/MM equilibration ───────────────────────────────────────────
stage="05_QMMM_restraint_free_simulations"
folder_stage="$DIR/$stage"
echo "Setting up $folder_stage"

sed \
    -e "s|PARM|$parm|g" \
    -e "s|GEOM|$geom|g" \
    -e "s|SCHEME|$scheme|g" \
    "$templates/run_1_eq_template" > "$folder_stage/run_1_eq.cmd"

coorb=$(echo "scale=1; ($coor0 - 1)" | bc -l)
coora=$(echo "scale=1; ($coor0 + 1)" | bc -l)
apply_restr_sel "$templates/$restr_template" "$folder_stage/restr" 10 \
    -e "s/COOR/$coor0/g" \
    -e "s/VALI/${coorb}/g" \
    -e "s/VALF/${coora}/g"
apply_qm_sel "$templates/in_1_eq_free_template" "$folder_stage/in"

# ─── STAGES 06–07: scan and PMF windows ──────────────────────────────────────
for stage in 06_scan_umbrella_sampling 07_PMF_umbrella_sampling; do
    folder_stage="$DIR/$stage"
    for window in $(seq 1 "$windows"); do
        val=$(echo "scale=1; $coor0 + ($window - 1) * $scan_step" | bc -l)
        vali=$(echo "scale=1; ($val - 10)" | bc -l)
        valf=$(echo "scale=1; ($val + 10)" | bc -l)
        echo "  $folder_stage  window $window  coord=$val"

        if [ "$stage" = "06_scan_umbrella_sampling" ]; then
            prev=$(( window - 1 ))
            apply_qm_sel "$templates/in_2_restrained_template" \
                "$folder_stage/${window}.in" -e "s/STEP/$window/g"
            sed \
                -e "s/STEP/$window/g" \
                -e "s/PREV/$prev/g" \
                -e "s|PARM|$parm|g" \
                -e "s|SCHEME|$scheme|g" \
                "$templates/run_2_scan_template" > "$folder_stage/${window}.cmd"
            restr_force=300  # kcal/mol/Å² — tight for scan
        else
            for step in eq PMF; do
                apply_qm_sel "$templates/in_3_${step}_template" \
                    "$folder_stage/${step}_${window}.in" -e "s/STEP/$window/g"
            done
            sed \
                -e "s/STEP/$window/g" \
                -e "s|PARM|$parm|g" \
                -e "s|SCHEME|$scheme|g" \
                "$templates/run_3_PMF_template" > "$folder_stage/${window}.cmd"
            restr_force=100  # kcal/mol/Å² — softer for PMF production
        fi

        apply_restr_sel "$templates/$restr_template" "$folder_stage/${window}.restr" "$restr_force" \
            -e "s/COOR/$val/g" \
            -e "s/VALI/$vali/g" \
            -e "s/VALF/$valf/g"
    done
done

# ─── JOB SUBMISSION ──────────────────────────────────────────────────────────
echo "Submitting jobs from $DIR"

equil_id=$(sbatch --parsable \
    --chdir="$DIR/05_QMMM_restraint_free_simulations" \
    "$DIR/05_QMMM_restraint_free_simulations/run_1_eq.cmd")
echo "  equilibration job id: $equil_id"

for window in $(seq 1 "$windows"); do
    if [ "$window" -eq 1 ]; then
        scan_id=$(sbatch --parsable \
            --chdir="$DIR/06_scan_umbrella_sampling" \
            --dependency=afterok:${equil_id} \
            "$DIR/06_scan_umbrella_sampling/${window}.cmd")
    else
        scan_id=$(sbatch --parsable \
            --chdir="$DIR/06_scan_umbrella_sampling" \
            --dependency=afterok:${scan_id} \
            "$DIR/06_scan_umbrella_sampling/${window}.cmd")
    fi
    sbatch \
        --chdir="$DIR/07_PMF_umbrella_sampling" \
        --dependency=afterok:${scan_id} \
        "$DIR/07_PMF_umbrella_sampling/${window}.cmd"
    echo "  window $window: scan_id=$scan_id"
done
