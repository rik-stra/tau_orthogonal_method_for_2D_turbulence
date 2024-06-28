# Description: This script trains a CNN model 
import sys
import os
import json
from aux_code.CNN_trainer import train_new_CNN

# read provided input file
input_file_path = sys.argv[1]
run_n = sys.argv[2]
with open(input_file_path, 'r') as f:
    input_dict = json.load(f)
# check all required keys are present
required_keys = ["model_dir", "train_file", "n_train_samples", "n_epochs", "add_rotations", "hist_directory", "CNN_id"]
for key in required_keys:
    if key not in input_dict:
        raise ValueError(f"Key {key} is missing from input file")

if os.path.exists(input_dict['hist_directory']) == False:
        os.makedirs(input_dict['hist_directory'])
hist_file = input_dict['hist_directory'] + '/hist_CNN_'+str(input_dict["CNN_id"])+'_run_'+str(run_n)+'.json'
model_dir = input_dict['model_dir'] + '_run_'+str(run_n)
# train the CNN model
train_new_CNN(train_file=input_dict["train_file"],
              model_dir=model_dir, 
              hist_file=hist_file,
              n_train_samples=input_dict["n_train_samples"],
              n_epochs= input_dict["n_epochs"],
              add_rotations= input_dict["add_rotations"])