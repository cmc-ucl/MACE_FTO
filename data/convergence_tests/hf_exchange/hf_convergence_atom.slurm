#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# Replace [budget code] below with your full project code
#SBATCH --account=e05-algor-smw
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --export=none

module load epcc-job-env
module load other-software
module load crystal

# Address the memory leak
export FI_MR_CACHE_MAX_COUNT=0
export SLURM_CPU_FREQ_REQ=2250000

# Run calculations
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_0HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_5HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_10HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_15HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_20HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Al_25HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_0HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_5HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_10HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_15HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_20HF
timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm Ga_25HF