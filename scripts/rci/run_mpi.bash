#!/bin/bash
#SBATCH --time=01:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --mem=64G
#SBATCH --partition=amd
#SBATCH --job-name=dim_mpi
#SBATCH --err=tmp/cg_dim.err
#SBATCH --out=tmp/cg_dim.out

ml foss/2021a
dim_dir="/home/rurabori/code/diploma_research"
matrix_name="it-2004"
impl="petsc"

srun --cpu-bind=cores --partition=amd --mpi=pmix $dim_dir/build/benchmarks/conjugate_gradient/$impl/RelWithDebInfo/cg_$impl -matrix_file "$dim_dir/resources/matrices/$matrix_name/$matrix_name.csr.h5"
