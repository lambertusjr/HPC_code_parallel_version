#!/bin/bash
#PBS -N build_RP_env
#PBS -l select=1:ncpus=1:mem=64GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o build_env.out
#PBS -V

set -euo pipefail
umask 0077
cd "$PBS_O_WORKDIR"

TMP="/scratch-small-local/${PBS_JOBID//./-}"
mkdir -p "$TMP"

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj /bin/micromamba
export MAMBA_ROOT_PREFIX="$TMP/micromamba"

# 1) create target env (do NOT upgrade pip here)
./bin/micromamba create -y --no-rc -p "$TMP/RP_env" -f environment.yaml

# 2) separate packer env with conda-pack
./bin/micromamba create -y --no-rc -p "$TMP/packer" -c conda-forge python=3.10 conda-pack

# 3) pack target using packer
./bin/micromamba run -p "$TMP/packer" conda-pack -p "$TMP/RP_env" -o "$PBS_O_WORKDIR/RP_env.tar.gz"

# cleanup on success
if [ -s "$PBS_O_WORKDIR/RP_env.tar.gz" ]; then
  /bin/rm -rf "$TMP"
fi

echo "Packed -> $PBS_O_WORKDIR/RP_env.tar.gz"
