#!/bin/bash
#PBS -N build_RP_env
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o build_env.out
#PBS -V

# 1. Enable 'x' to print commands as they execute
set -euxo pipefail

# 2. Trap errors to print the line number of the failure
trap 'echo ">>> ERROR: Script crashed on line $LINENO"' ERR

echo ">>> JOB START: $PBS_JOBID"
echo ">>> DATE: $(date)"

umask 0077
cd "$PBS_O_WORKDIR"
echo ">>> WORKDIR: $(pwd)"

# 3. Check if input file exists before starting
if [ ! -f "environment.yaml" ]; then
    echo ">>> ERROR: environment.yaml is missing in $PBS_O_WORKDIR"
    ls -l
    exit 1
fi

# 4. Check if the binary exists on the login node (Pre-req check)
if [ ! -f "bin/micromamba" ]; then
    echo ">>> ERROR: bin/micromamba not found in $PBS_O_WORKDIR"
    echo ">>> Please run the curl download command on the login node first."
    exit 1
fi

SCRATCH_BASE="/scratch-small-local"
# Check if scratch dir exists AND is writable. If not, fallback to home scratch.
if [ ! -w "${SCRATCH_BASE}" ]; then
    echo ">>> /scratch-small-local not writable or missing. Falling back to \$HOME/scratch"
    SCRATCH_BASE="$HOME/scratch"
fi

SPACED="${PBS_JOBID//./-}" 
TMP=/scratch-small-local/${SPACED} # E.g. 249926.hpc1.hpc
mkdir -p ${TMP}
echo "Temporary work dir: ${TMP}"

echo ">>> STEP 1: Copying Micromamba to Scratch..."
# Copy the binary from the submit dir to the compute node scratch
# This prevents network issues and speeds up execution
cp "$PBS_O_WORKDIR/bin/micromamba" "$TMP/micromamba"
chmod +x "$TMP/micromamba"

# Define the path to the executable in TMP
MAMBA_EXE="$TMP/micromamba"

# Debug: Check if binary actually exists in TMP
if [ ! -f "$MAMBA_EXE" ]; then
    echo ">>> ERROR: Micromamba binary not found in TMP after copy."
    ls -l "$TMP"
    exit 1
fi

export MAMBA_ROOT_PREFIX="$TMP/micromamba_root"

echo ">>> STEP 2: Creating Target Environment..."
$MAMBA_EXE create -y --no-rc -p "$TMP/RP_env" -f environment.yaml

echo ">>> STEP 2.1: Installing PyTorch Geometric Stack..."
RP_PIP="$TMP/RP_env/bin/pip"

# Verify pip exists
if [ ! -f "$RP_PIP" ]; then
    echo ">>> ERROR: pip binary not found at $RP_PIP"
    exit 1
fi

# Explicit install command - this is much safer than YAML
$RP_PIP install torch_geometric
$RP_PIP install torch_scatter torch_sparse torch_cluster torch_spline_conv pyg_lib \
    -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

echo ">>> STEP 3: Creating Packer Environment..."
$MAMBA_EXE create -y --no-rc -p "$TMP/packer" -c conda-forge python=3.10 conda-pack

echo ">>> STEP 4: Packing Environment..."
# Verify packer env exists
if [ ! -f "$TMP/packer/bin/conda-pack" ]; then
    echo ">>> ERROR: conda-pack binary missing in packer env"
    ls -l "$TMP/packer/bin/"
    exit 1
fi

$MAMBA_EXE run -p "$TMP/packer" conda-pack -p "$TMP/RP_env" -o "$PBS_O_WORKDIR/RP_env.tar.gz"

# Cleanup
echo ">>> STEP 5: Verifying Output..."
if [ -s "$PBS_O_WORKDIR/RP_env.tar.gz" ]; then
  echo ">>> SUCCESS: File created at $PBS_O_WORKDIR/RP_env.tar.gz"
  /bin/rm -rf "$TMP"
else
  echo ">>> FAILURE: Output file missing or empty."
  exit 1
fi