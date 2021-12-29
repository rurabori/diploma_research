#!/bin/bash
#SBATCH --time=01:00:00

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --mem=128G
#SBATCH --partition=amd
#SBATCH --job-name=petsc_mpi
#SBATCH --err=tmp/cg_petsc.err
#SBATCH --out=tmp/cg_petsc.out

ml foss/2021a
dim_dir="/home/rurabori/code/diploma_research"
matrix_name="GAP-kron"

output_dir="$dim_dir/tmp/cg_petsc/$SLURM_JOB_ID"

mkdir -p $output_dir
cd $output_dir

matrix_file_bee="/mnt/beegfs/cpu/temporary/$matrix_name.csr.h5"

if [[ ! -f "$matrix_file_bee" ]]; then
    echo "file $matrix_file_bee doesn't exist, copy it to beegfs" && exit 1
fi

srun --cpu-bind=cores --partition=amd --mpi=pmix $dim_dir/build/benchmarks/conjugate_gradient/petsc/Release/cg_petsc -matrix_file "$matrix_file_bee"
