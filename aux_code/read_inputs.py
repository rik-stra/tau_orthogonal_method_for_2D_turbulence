""" 
This file contains functions to read and modify input files:
 -  Inputs class: reads the input file for simulation
 -  adapt_input_file: adapts the flags of a base file and saves as a new file
author: Rik Hoekstra (28-6-2024)"""

import json
from aux_code.filters import Grids, Filters

class Inputs:
    """
    Class to read the input file for simulation.
    """

    ##### mandatory input flags  #####
    # N_HF, N_LF, remove_alias, use_gaussian_filter, t_start, simulation_time, dt_HF
    
    ##### define all optional input flags  ##########################      
    decay_time_nu = 5.0; decay_time_mu = 90.0
    time_integration_scheme = "RK4"
    sim_ID = "none"; adapt_nu_to_LF = False; 
    store_qoi_trajectories = True; store_final_state=False;
    restart = True; restart_file_name = "none"
    store_frame_rate = 10
    
    ### only needed for low fidelity simulations ####
    parametrization = 'none'
    parametrization_file_path='none'; discretization_SGS_term=1
        # for TO_track
    ref_file_permutation= 'id'; dQ_discretization = "np1"; relax_constant = 1.0
        # for TO
    sample_independent = False; sample_domain = "none"; use_MVG = False
        # for smag
    smag_constant = 0.1

    ### only needed for high fidelity simulations ####
    create_training_data = False
    
    
    def __init__(self, filename, run_number=0):
        """
        Initialize the Inputs object.

        Parameters:
        - filename (str): The path to the input file.
        - run_number (int): The run number for the simulation (default: 0).
        """
        with open(filename, 'r') as fp:

            #print the desription of the input file
            print(fp.readline())
            varia = json.loads(fp.readline())

            ################################ set up grid and projection operators ################################################
            N = varia["N_HF"]
            N_LF = varia["N_LF"]
            remove_alias = varia["remove_alias"]
            print("GRID:")
            print("reference: N=", N, "  Coarse: N_LF=", N_LF)
            if remove_alias:
                N_a = (N-1)*3//2+1
                N_LF_a = (N_LF-1)*3//2+1
            else:
                N_a=N; N_LF_a=N_LF
            
            self.grid = Grids(N_LF_a,N_a, N_LF_resolved=N_LF, N_HF_resolved=N)
            self.filters = Filters(self.grid, remove_alias, varia["use_gaussian_filter"])
            
            ######################################################################################################################
            self.t_start = varia["t_start"]; self.simulation_time = varia["simulation_time"]; self.dt_HF = varia["dt_HF"]
            print("TIME SETTINGS: (in days)")
            print("T start:", varia["t_start"], "T end:", varia["t_start"]+varia["simulation_time"], "dt_HF:", varia["dt_HF"])


            #########################
            # Read simulation flags #
            #########################
           
            ##### read from json file  ####################################################
            flags_line = fp.readline()[:-1]
            while flags_line[-1] != '}':
                flags_line = flags_line+fp.readline()[:-1]
            flags = json.loads(flags_line)
            print('*********************')
            print('Simulation flags')
            for key in flags.keys():
                vars(self)[key] = flags[key]
                print(key, '=', flags[key])
            
            ## read QoI to track (or correct)
            self.N_Q_track = int(fp.readline())
            self.targets = []; self.V = []; self.T_is = []
            for i in range(self.N_Q_track):
                qoi_i = json.loads(fp.readline())
                self.targets.append(qoi_i['target'])
                self.V.append(qoi_i['V_i'])
                self.T_is.append(qoi_i['V_i'])
                k_min = qoi_i['k_min']
                k_max = qoi_i['k_max']
                #use targeted spectral filters, could differ per QoI
                self.filters.initialize_P_i(k_min, k_max)
            
            ## read QoI to store
            self.N_Q_save = int(fp.readline())
            for j in range(self.N_Q_save):
                qoi_i = json.loads(fp.readline())
                self.targets.append(qoi_i['target'])
                
                k_min = qoi_i['k_min']
                k_max = qoi_i['k_max']
                #use targeted spectral filters, could differ per QoI
                self.filters.initialize_P_i(k_min, k_max)

            fp.close()
            print('*********************')

            if run_number > 0:
                self.sim_ID = self.sim_ID + "_run_" + str(run_number)
                if self.parametrization == "CNN":
                    self.parametrization_file_path = self.parametrization_file_path + "_run_" + str(run_number)
            #####################################################################################


def adapt_input_file(base_file, new_file, new_flags):
    """
    Adapt the flags of a base file and save as a new file. Note that the base file should contain all flags that are adjusted.

    Parameters:
    - base_file (str): The path to the base input file.
    - new_file (str): The path to the new input file.
    - new_flags (dict): A dictionary containing the new flags to be updated.

    Returns:
    - int: 0 if successful, 1 if key not found in base file.
    """
    #read base file
    with open(base_file, 'r') as fp:
        fp.readline() # skip first line
        base_dict_1 = json.loads(fp.readline()[:-1])
        flags_line = fp.readline()[:-1]
        while flags_line[-1] != '}':
            flags_line = flags_line+fp.readline()[:-1]
        base_dict_2 = json.loads(flags_line) 
        base_num_qoi = int(json.loads(fp.readline()))
        base_qoi_dicts = []
        for i in range(base_num_qoi):
            base_qoi_dicts.append(json.loads(fp.readline()))
        base_num_track = int(json.loads(fp.readline()))
        base_track_dicts = []
        for i in range(base_num_track):
            base_track_dicts.append(json.loads(fp.readline()))
    
    #update flags
    for key in new_flags.keys():
        if key in base_dict_1.keys():
            base_dict_1[key] = new_flags[key]
        elif key in base_dict_2.keys():
            base_dict_2[key] = new_flags[key]
        else:
            print('key not found in base file:', key)
            return 1
            

    #write new file
    with open(new_file, 'w') as fp:
        fp.write('Grid Search Input File \n')
        json.dump(base_dict_1, fp)
        fp.write('\n')
        json.dump(base_dict_2, fp)
        fp.write('\n')
        fp.write(str(base_num_qoi)+'\n')
        for i in range(base_num_qoi):
            json.dump(base_qoi_dicts[i], fp)
            fp.write('\n')
        fp.write(str(base_num_track)+'\n')
        for i in range(base_num_track):
            json.dump(base_track_dicts[i], fp)
            fp.write('\n')
    return 0