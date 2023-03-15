#!/bin/sh

#SBATCH -J ICSolver
#SBATCH -o log_%j.txt
#SBATCH -e log_%j.err
#SBATCH -p  mpi
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
####SBATCH --constraint="epyc"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export RUNDIR=./


export CHOMBO_HOME=/home/cjoana/git/Chombo/lib
export GRCHOMBO_SOURCE=/home/cjoana/git/GRChombo/Source


srun $RUNDIR/Main_PoissonSolver3d.Linux.64.mpic++.gfortran.OPTHIGH.MPI.OPENMPCC.ex ./ics_params.txt
# Main_ScalarField3d.Linux.64.mpic++.gfortran.OPT.MPI.OPENMPCC.ex params_L5.txt

###SBATCH --mem-per-cpu=100


####SBATCH -J myjobname
#####SBATCH -o log_%j.txt
#####SBATCH -e log_%j.err
#####SBATCH -p mpi
#####SBATCH --time=00:05:00
#####SBATCH --ntasks=3
#####SBATCH --cpus-per-task=2
#####SBATCH --constraint="xeon"
#####SBATCH --time=2:02:00
####export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
####cd /home/myname
####srun ./myprogram
