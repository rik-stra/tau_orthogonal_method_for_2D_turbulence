#!/bin/bash
#SBATCH -J smag_array_job
#SBATCH --array=1-14
#SBATCH -t 25:00
#SBATCH -p gpu
#SBATCH --partition=gpu_mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1


run_n=$SLURM_ARRAY_TASK_ID
sim_folder=rk4_new

# call with 

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a

# copy data to scratch
mkdir -p $TMPDIR/pre_computed_data/restart
cp -r $HOME/$sim_folder/pre_computed_data/restart/get_initial_conditions_t_300.0.hdf5 $TMPDIR/pre_computed_data/restart/.

cp $HOME/$sim_folder/inputs/gridsearch/smag/in_smag_$run_n.json $TMPDIR
# create dirs for output
mkdir $TMPDIR/output
mkdir $TMPDIR/output/samples
mkdir $TMPDIR/output/samples/gridsearch
# make scratch the current directory
cd $TMPDIR
python $HOME/$sim_folder/LF_simulation.py $TMPDIR/in_smag_$run_n.json
# copy data from scratch to home
mkdir -p $HOME/$sim_folder/output/samples/gridsearch
cp $TMPDIR/output/samples/gridsearch/* $HOME/$sim_folder/output/samples/gridsearch/.