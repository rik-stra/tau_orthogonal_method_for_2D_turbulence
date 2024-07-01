''' Description:
This script trains a CNN model using parameters provided in a json file. As second input the runnumber should be provided (this allows creating replicas of "the same CNN").
In the json file, the following keys must be present:
- model_dir: directory where the model will be saved
- CNN_id: id of the CNN model will be combined with the run number to create the file names. 
- train_file: path to the training data
- n_train_samples: number of training samples
- n_epochs: number of epochs
- add_rotations: boolean, whether to add rotations to the training data
- hist_directory: directory where the training history will be saved (file is called CNN_id.json)
'''
import sys
import os
import json
from aux_code.CNN_trainer import train_new_CNN

### read provided input file ###
input_file_path = sys.argv[1]
run_n = sys.argv[2]
with open(input_file_path, 'r') as f:
    input_dict = json.load(f)

### check all required keys are present  ###
required_keys = ["model_dir", "train_file", "n_train_samples", "n_epochs", "add_rotations", "hist_directory", "CNN_id"]
for key in required_keys:
    if key not in input_dict:
        raise ValueError(f"Key {key} is missing from input file")

### create directories if they do not exist ###
if os.path.exists(input_dict['hist_directory']) == False:
        os.makedirs(input_dict['hist_directory'])
hist_file = input_dict['hist_directory'] + '/hist_CNN_'+str(input_dict["CNN_id"])+'_run_'+str(run_n)+'.json'
model_dir = input_dict['model_dir'] + '_run_'+str(run_n)

### train the CNN model ###
train_new_CNN(train_file=input_dict["train_file"],
              model_dir=model_dir, 
              hist_file=hist_file,
              n_train_samples=input_dict["n_train_samples"],
              n_epochs= input_dict["n_epochs"],
              add_rotations= input_dict["add_rotations"])