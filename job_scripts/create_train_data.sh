#!/bin/bash
#SBATCH -J create_train_data
#SBATCH -t 120:00
#SBATCH -p gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1

sim_folder=create_training_data
# call with 

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a

# copy data to scratch
mkdir -p $TMPDIR/pre_computed_data/restart
cp -r $HOME/$sim_folder/pre_computed_data/restart/get_initial_conditions_t_300.0.hdf5 $TMPDIR/pre_computed_data/restart/.

# create dirs for output
mkdir $TMPDIR/output
mkdir $TMPDIR/output/samples
# make scratch the current directory
cd $TMPDIR
python $HOME/$sim_folder/compute_reference.py $HOME/$sim_folder/inputs/in_get_training_data_gauss.json
mkdir -p $HOME/$sim_folder/output
cp -r $TMPDIR/output/* $HOME/$sim_folder/output/.

