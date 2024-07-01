"""
========================================================================
python script:
Script to perform a high fidelity simulation of a 2D turbulent flow using a pseudo-spectral method.
Can be used to create training data sets in the form of:
    - QoI trajectories for the TO method
    - Filtered input and target fields for the CNN parametrization

Input: 
    - json file containing the input flags for the simulation
    - run_number: integer, optional, for multiple runs with the same input file

Output: (depending on input flags)
    - hdf5 file containing the QoI trajectories
    - hdf5 file containing the training data for the CNN
    - hdf5 file containing the final state of the system (for restarts)

Author: R. Hoekstra (1-7-2024)
========================================================================
"""
import numpy as np
import torch
import os
import sys
import time

from aux_code.functions_for_solver import *
from aux_code.plot_store_functions import store_samples_hdf5, store_training_data_hdf5, store_state
from aux_code.read_inputs import Inputs
from aux_code.time_integrators import Solution_state


###########################
# M A I N   P R O G R A M #
###########################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HOME = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(HOME + '/output') == False:
        os.makedirs(HOME + '/output')

#read flags from input file
input_file_path = sys.argv[1]
if len(sys.argv) ==3:
    run_number = int(sys.argv[2])
else: run_number = 0

#read inputs
input = Inputs(input_file_path, run_number)

################################ set up grid and projection operators ################################################
grid = input.grid
filters = input.filters

##############  Time integrations settings  ##############
#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

dt = input.dt_HF*day
t = input.t_start*day
t_end = t + input.simulation_time*day
n_steps = int(np.round((t_end-t)/dt))

######### Set up parameters which depend on input flags ############################
nu = 1.0/(day*(256//3)**2*input.decay_time_nu)
if input.adapt_nu_to_LF :   # hyper viscosity
    nu_LF = 1.0/(day*filters.Ncutoff_LF**2*input.decay_time_nu)
else:
    nu_LF = nu
mu = 1.0/(day*input.decay_time_mu)


###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#framerate of storing data, plotting results (1 = every integration time step)
steps_per_day = np.floor(day/dt).astype('int')
store_frame_rate = 10                                 # every 10th time step we store the QoIs
plot_frame_rate = np.floor(input.store_frame_rate*day/dt).astype('int')  # every 100th day we store the solution
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')+1
S_fields = np.floor(n_steps/plot_frame_rate).astype('int')+2

#TRAINING DATA SET
QoI = ['Q_HF', 'w_hat_n_HF_filtered'] #'c_ij',,'laplacian_hat', 'inner_prods', 'src_Q', 'EF_hat_exact'


#allocate memory
if input.store_qoi_trajectories == True:
    samples = {}

    QoI_fields = []
    QoI_scalars = []
    samples['S'] = S
    samples['N_LF'] = grid.N_LF_resolved
    for q in range(len(QoI)):
        #assume a field contains the string '_hat_'
        if 'w_hat_n_HF_filtered' in QoI[q]:
            QoI_fields.append(QoI[q])
            samples[QoI[q]] = np.zeros([S_fields, grid.N_LF_resolved, grid.N_LF_resolved]) + 0.0j
        elif 'w_hat_n_HF' in QoI[q]:
            QoI_fields.append(QoI[q])
            samples[QoI[q]] = np.zeros([S_fields, grid.N_HF_resolved, grid.N_HF_resolved]) + 0.0j
        #a scalar
        else:
            QoI_scalars.append(QoI[q])
            samples[QoI[q]] = np.zeros([S,input.N_Q_save])

if input.create_training_data:
    n_training_samples = (n_steps-20)//steps_per_day+1
    training_data = {}
    training_data["input_w_n"] = np.zeros([n_training_samples, grid.N_LF_resolved, grid.N_LF_resolved])
    training_data["input_w_nm1"] = np.zeros([n_training_samples, grid.N_LF_resolved, grid.N_LF_resolved])
    training_data["input_w_np1"] = np.zeros([n_training_samples, grid.N_LF_resolved, grid.N_LF_resolved])
    training_data["r_bar"] = np.zeros([n_training_samples, grid.N_LF_resolved, grid.N_LF_resolved])             # renamed was: out_r_filtered
    training_data["r_unrolled"] = np.zeros([n_training_samples, grid.N_LF_resolved, grid.N_LF_resolved])        # renamed was: out_r
    training_data["Q_HF_LF"] = np.zeros([n_training_samples, 2, input.N_Q_save])

#######  forcing term   ########################
F_LF = 2**1.5*np.cos(5*grid.x_LF)*np.cos(5*grid.y_LF)
F_hat_LF = np.fft.fft2(F_LF)
F_hat_HF = np.zeros((grid.N_HF,grid.N_HF))+0.0j
F_hat_HF[filters.P_HF2LF==1] = F_hat_LF[filters.P_LF==1]*(grid.N_HF/grid.N_LF)**2
if filters.use_gaussian_filter: F_hat_LF *= filters.gaussian_filter
##################################################

#########  inital conditions ######################
solution_state_HF = Solution_state(input.time_integration_scheme, t, dt, nu, mu, grid, "HF")
solution_state_HF.intialize_state(input, filters, grid, device)
if input.create_training_data:
    solution_state_LF = Solution_state(input.time_integration_scheme, t, dt, nu_LF, mu, grid, "LF")
    solution_state_LF.intialize_state(input, filters, grid, device)

####################################################

######### put everything on the GPU ######
# fields
F_hat_HF= torch.from_numpy(F_hat_HF).to(device=device)

# fields LF
F_hat_LF= torch.from_numpy(F_hat_LF).to(device=device)

# objects
filters.P_HF = torch.from_numpy(filters.P_HF).to(device=device)
filters.P_LF = torch.from_numpy(filters.P_LF).to(device=device)
filters.P_HF2LF = torch.from_numpy(filters.P_HF2LF).to(device=device)
if filters.use_gaussian_filter: filters.gaussian_filter = torch.from_numpy(filters.gaussian_filter).to(device=device)

grid.k_x_HF = torch.from_numpy(grid.k_x_HF).to(device=device)
grid.k_y_HF = torch.from_numpy(grid.k_y_HF).to(device=device)
grid.k_squared_nonzero_HF = torch.from_numpy(grid.k_squared_nonzero_HF).to(device=device)
grid.k_squared_HF = torch.from_numpy(grid.k_squared_HF).to(device=device)

grid.k_x_LF = torch.from_numpy(grid.k_x_LF).to(device=device)
grid.k_y_LF = torch.from_numpy(grid.k_y_LF).to(device=device)
grid.k_squared_nonzero_LF = torch.from_numpy(grid.k_squared_nonzero_LF).to(device=device)
grid.k_squared_LF = torch.from_numpy(grid.k_squared_LF).to(device=device)

filters.move_Pi_to_GPU(device)

solution_state_HF.set_grid_and_filter(grid, filters)
if input.create_training_data:
    solution_state_LF.set_grid_and_filter(grid, filters)


print('*********************')
print('Solving forced dissipative vorticity equations on '+ str(device))
print('Ref grid = ', grid.N_HF_resolved, 'x', grid.N_HF_resolved)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')
print('*********************')

t0 = time.time()


############ time loop  ########################
#some counters
j = 0
j_low_fidelity = 0
Q_HF = np.zeros(input.N_Q_save)

for n in range(n_steps):
    
    if np.mod(n, int(day/dt)) == 0:
        print(f'day =  {n//int(day/dt)} of  {n_steps//int(day/dt)}')
    
    ########run the HF model#####################################################################    
    #solve for next time step
    solution_state_HF.time_step(F_hat_HF)
            
    ######## once per day take some LF time steps to create training data #####
    if input.create_training_data:
        ##### 1. get LF state at t-dt_LF ####
        if np.mod(n,steps_per_day) == 0:
            solution_state_LF.w_hat_nm1 = filters.filter_HF2LF_torch(solution_state_HF.w_hat_n)
            solution_state_LF.psi_hat_nm1 = filters.filter_HF2LF_torch(solution_state_HF.psi_hat_n)
            if input.time_integration_scheme in ["AB/BDI2","AB/CN"]:
                solution_state_LF.VgradW_hat_nm1 = filters.filter_HF2LF_torch(solution_state_HF.VgradW_hat_n)

            training_data['input_w_nm1'][j_low_fidelity,:,:] = torch.fft.ifft2(filters.filter_LF2resolved(solution_state_LF.w_hat_n)).real.cpu()
        ###### 2. get LF state at t  #####
        if np.mod(n-10,steps_per_day) == 0:
            solution_state_LF.w_hat_n = filters.filter_HF2LF_torch(solution_state_HF.w_hat_n)
            solution_state_LF.psi_hat_n = filters.filter_HF2LF_torch(solution_state_HF.psi_hat_n)

            training_data['input_w_n'][j_low_fidelity,:,:] = torch.fft.ifft2(filters.filter_LF2resolved(solution_state_LF.w_hat_n)).real.cpu()

            ### 2b. take one LF time step ###
            solution_state_LF.time_step(F_hat_LF)
            
            training_data['input_w_np1'][j_low_fidelity,:,:] = torch.fft.ifft2(filters.filter_LF2resolved(solution_state_LF.w_hat_np1)).real.cpu()

            ## compute commutator error from HF solution  ##
            if input.time_integration_scheme in ["RK4"]:
                solution_state_HF.VgradW_hat_n = compute_VgradW_hat(solution_state_HF.w_hat_n, solution_state_HF.psi_hat_n, solution_state_HF.P, solution_state_HF.k_x, solution_state_HF.k_y)
                solution_state_LF.VgradW_hat_n = compute_VgradW_hat(solution_state_LF.w_hat_n, solution_state_LF.psi_hat_n, solution_state_LF.P, solution_state_LF.k_x, solution_state_LF.k_y)
            r_filtered = solution_state_LF.VgradW_hat_n-filters.filter_HF2LF_torch(solution_state_HF.VgradW_hat_n)
            training_data['r_bar'][j_low_fidelity,:,:] = torch.fft.ifft2(filters.filter_LF2resolved(r_filtered)).real.cpu()

            ## compute Q_LF_{n+1}
            training_data["Q_HF_LF"][j_low_fidelity, 1, :] = get_QLF(input.targets,N_Q=input.N_Q_save, w_hat_LF=solution_state_LF.w_hat_np1, psi_hat_LF=solution_state_LF.psi_hat_np1, filters=filters, device=device).cpu()

        ####### ten high fidelity time steps later, the reference is known and we can calculate the subgrid term based on 1 step unrolling####
        if np.mod(n-20,steps_per_day) ==0:
            r_unrolled = -(filters.filter_HF2LF_torch(solution_state_HF.w_hat_n) - solution_state_LF.w_hat_np1)/solution_state_LF.norm_factor
            training_data['r_unrolled'][j_low_fidelity,:,:] = torch.fft.ifft2(filters.filter_LF2resolved(r_unrolled)).real.cpu()
            
            for i in range(input.N_Q_save):
                training_data["Q_HF_LF"][j_low_fidelity, 0, i] =  np.real(get_qoi(filters.apply_P_i(solution_state_HF.w_hat_n, i), filters.apply_P_i(solution_state_HF.psi_hat_n,i), 
                            input.targets[i], grid.N_HF))
            j_low_fidelity+=1
    
    ###  Compute & store QoI to dictionary ###
    if input.store_qoi_trajectories == True:
        #calculate the QoI
        if np.mod(n,store_frame_rate) == 0:
            for i in range(input.N_Q_save):
                Q_HF[i] = np.real(get_qoi(filters.apply_P_i(solution_state_HF.w_hat_n, i), filters.apply_P_i(solution_state_HF.psi_hat_n,i), 
                                input.targets[i], grid.N_HF))
            for qoi in QoI_scalars:
                samples[qoi][n//store_frame_rate] = eval(qoi)
        # store fields (once every plot_frame_rate)
        if np.mod(n,plot_frame_rate) == 0:
            w_hat_n_HF_filtered = filters.filter_HF2LF_torch(solution_state_HF.w_hat_n)
            samples['w_hat_n_HF_filtered'][j] = filters.filter_LF2resolved(w_hat_n_HF_filtered).cpu()
            j += 1
    
    ### store data for restart ###
    if (n==n_steps-10):
        w_hat_nm10_HF=filters.filter_HF2resolved(solution_state_HF.w_hat_n).cpu().numpy(force=True)
        if input.time_integration_scheme in ["RK4"]:
            solution_state_HF.VgradW_hat_n = compute_VgradW_hat(solution_state_HF.w_hat_n, solution_state_HF.psi_hat_n, solution_state_HF.P, solution_state_HF.k_x, solution_state_HF.k_y)
        VgradW_hat_nm10_HF = filters.filter_HF2resolved(solution_state_HF.VgradW_hat_n).cpu().numpy(force=True)
    
    if (n==n_steps-1):
        w_hat_nm1_HF=filters.filter_HF2resolved(solution_state_HF.w_hat_n).cpu().numpy(force=True)
        if input.time_integration_scheme in ["RK4"]:
            solution_state_HF.VgradW_hat_n = compute_VgradW_hat(solution_state_HF.w_hat_n, solution_state_HF.psi_hat_n, solution_state_HF.P, solution_state_HF.k_x, solution_state_HF.k_y)
        VgradW_hat_nm1_HF = filters.filter_HF2resolved(solution_state_HF.VgradW_hat_n).cpu().numpy(force=True)

    #update variables
    solution_state_HF.update_vars()
    
    
    
    ## check for nans ###########
    if torch.isnan(torch.sum(solution_state_HF.w_hat_n)):
        print('nan detected in solution, stopping simulation')
        break
##### end of time loop #########################################################################################################
t1 = time.time()
print('Simulation time =', t1 - t0, 'seconds')

if input.store_qoi_trajectories == True:
    # store last fields 
    w_hat_n_HF_filtered = filters.filter_HF2LF_torch(solution_state_HF.w_hat_n)
    samples['w_hat_n_HF_filtered'][j] = filters.filter_LF2resolved(w_hat_n_HF_filtered).cpu()

#store the samples
if input.store_qoi_trajectories == True: store_samples_hdf5(HOME,input.sim_ID,solution_state_HF.t/day, QoI, samples)
#store the training data
if input.create_training_data: store_training_data_hdf5(HOME,input.sim_ID,solution_state_HF.t/day, training_data)    

#store the state of the system to allow for a simulation restart at t > 0
if input.store_final_state == True:
    restart_dic = {'w_hat_nm1_HF': w_hat_nm1_HF,
                    'w_hat_n_HF': filters.filter_HF2resolved(solution_state_HF.w_hat_n).cpu().numpy(force=True), 
                    'VgradW_hat_nm1_HF': VgradW_hat_nm1_HF,
                    'w_hat_nm10_HF':w_hat_nm10_HF, 'VgradW_hat_nm10_HF':VgradW_hat_nm10_HF}
    store_state(restart_dic, HOME, input.sim_ID, solution_state_HF.t/day)


