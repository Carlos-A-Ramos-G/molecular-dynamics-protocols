#!/bin/sh
#
# Method 1: setup for a fully dual-topology side chain residue
#

tleap=$AMBERHOME/bin/tleap
basedir=leap
params=$(dirname "$(readlink -f "$0")")

$tleap -f - <<_EOF
# load the AMBER force fields
source leaprc.protein.ff14SB
source leaprc.water.tip3p

loadamberprep   $params/FCE.prepin
loadAmberParams $params/FCE_ff14SB.frcmod
loadAmberParams $params/FCE_gaff.frcmod

loadamberprep   $params/LTL.prepin
loadAmberParams $params/LTL_ff14SB.frcmod
loadAmberParams $params/LTL_gaff.frcmod

loadamberprep   $params/GQU.prepin
loadAmberParams $params/GQU_ff14SB.frcmod
loadAmberParams $params/GQU_gaff.frcmod

loadamberprep   $params/LCN.prepin
loadAmberParams $params/LCN_ff14SB.frcmod
loadAmberParams $params/LCN_gaff.frcmod

complex = loadpdb complex_solvated_raw.pdb
ligands = loadpdb ligands.pdb

set complex box {94.2564669 103.2504730  99.4386433}

solvatebox ligands TIP3PBOX 12

saveamberparm complex complex_solvated.prmtop complex_solvated.inpcrd
savepdb complex complex_solvated.pdb

saveamberparm ligands  ligands_solvated.prmtop ligands_solvated.inpcrd
savepdb ligands ligands_solvated.pdb

quit
_EOF
