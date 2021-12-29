#!/bin/bash
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --mem=128G
#SBATCH --partition=amd
#SBATCH --job-name=cg_dim_sp
#SBATCH --err=tmp/cg_dim_sp2.err
#SBATCH --out=tmp/cg_dim_sp2.out

ml foss/2021a
dim_dir="/home/rurabori/code/diploma_research"
matrix_name="GAP-web"

output_dir="$dim_dir/tmp/cg_dim_sp/$SLURM_JOB_ID"

mkdir -p $output_dir
cd $output_dir

matrix_file_bee=""
matrix_file="/data/temporary/$matrix_name.csr5.h5"

cp "$dim_dir/resources/matrices/$matrix_name/$matrix_name.csr5.h5" "$matrix_file"

srun --cpu-bind=cores --partition=amd $dim_dir/build/benchmarks/conjugate_gradient/dim_sp/Release/dim_sp "$matrix_file"
