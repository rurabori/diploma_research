#!/bin/bash
#SBATCH --time=01:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --mem=64G
#SBATCH --partition=amd
#SBATCH --job-name=dim_mpi
#SBATCH --err=tmp/cg_dim.err
#SBATCH --out=tmp/cg_dim.out

ml foss/2021a
dim_dir="/home/rurabori/code/diploma_research"
matrix_name="GAP-kron"

output_dir="$dim_dir/tmp/cg_dim/$SLURM_JOB_ID"

mkdir -p $output_dir
cd $output_dir

matrix_file_bee="/mnt/beegfs/cpu/temporary/$matrix_name.csr5.h5"

dim_binary="$dim_dir/build/benchmarks/conjugate_gradient/dim/Release/cg_dim"

srun --cpu-bind=cores --partition=amd --mpi=pmix "$dim_binary" "$matrix_file_bee"
