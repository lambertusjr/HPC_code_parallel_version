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

TMP="/scratch-small-local/${PBS_JOBID//./-}"
echo ">>> Creating TMP directory: $TMP"
mkdir -p "$TMP"

echo ">>> STEP 1: Downloading Micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvjf - bin/micromamba

# Debug: Check if binary actually exists after download
if [ ! -f "./bin/micromamba" ]; then
    echo ">>> ERROR: bin/micromamba not found after extraction."
    ls -R bin/
    exit 1
fi

export MAMBA_ROOT_PREFIX="$TMP/micromamba"

echo ">>> STEP 2: Creating Target Environment..."
./bin/micromamba create -y --no-rc -p "$TMP/RP_env" -f environment.yaml

echo ">>> STEP 3: Creating Packer Environment..."
./bin/micromamba create -y --no-rc -p "$TMP/packer" -c conda-forge python=3.10 conda-pack

echo ">>> STEP 4: Packing Environment..."
# Verify packer env exists
if [ ! -f "$TMP/packer/bin/conda-pack" ]; then
    echo ">>> ERROR: conda-pack binary missing in packer env"
    ls -l "$TMP/packer/bin/"
    exit 1
fi

./bin/micromamba run -p "$TMP/packer" conda-pack -p "$TMP/RP_env" -o "$PBS_O_WORKDIR/RP_env.tar.gz"

# Cleanup
echo ">>> STEP 5: Verifying Output..."
if [ -s "$PBS_O_WORKDIR/RP_env.tar.gz" ]; then
  echo ">>> SUCCESS: File created at $PBS_O_WORKDIR/RP_env.tar.gz"
  /bin/rm -rf "$TMP"
else
  echo ">>> FAILURE: Output file missing or empty."
  exit 1
fi