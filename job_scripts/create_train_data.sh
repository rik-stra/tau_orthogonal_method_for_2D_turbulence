#!/bin/bash
#SBATCH -J create_train_data
#SBATCH -t 360:00
#SBATCH -p gpu
#SBATCH --partition=gpu_mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1


# call with 

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a

# copy data to scratch
cp $HOME/create_train_data/get_initial_conditions_t_300.0.hdf5 $TMPDIR
# create dirs for output
mkdir $TMPDIR/output
mkdir $TMPDIR/output/samples
# make scratch the current directory
cd $TMPDIR
python $HOME/create_train_data/compute_reference.py $HOME/create_train_data/inputs/in_get_training_data_gauss.json
cp -r $TMPDIR/output $HOME/create_train_data/output
python $HOME/create_train_data/compute_reference.py $HOME/create_train_data/inputs/in_get_training_data.json
# copy data from scratch to home
cp -r $TMPDIR/output $HOME/create_train_data/output