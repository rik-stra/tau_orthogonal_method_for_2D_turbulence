"""
========================================================================
python script:
Script to perform a low fidelity simulation.
R. Hoekstra

========================================================================
"""

from aux_code.functions_for_solver import ReferenceFile_reader, get_w_hat_np1, get_psi_hat, get_QHF_QLF, evaluate_expression_simple, get_QLF
from aux_code.plot_store_functions import store_samples_hdf5, store_state
from aux_code.filters import Filters, Grids
from aux_code.read_inputs import Inputs
from aux_code.time_integrators import Solution_state

from aux_code.to_parametrization import Dq_sampler, TO_masks, reduced_r_fast
from aux_code.CNN_parametrization import Plain_CNN_surrogate
from aux_code.smagorinsky_parametrization import smag_parametrization

###########################
# M A I N   P R O G R A M #
###########################

import numpy as np
import torch
import os
import sys
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HOME = '.'
if os.path.exists(HOME+'/output') == False:
        os.makedirs(HOME+'/output')
########### Open input file ############################
#read flags from input file
input_file_path = sys.argv[1]
if len(sys.argv) ==3:
    run_number = int(sys.argv[2])
else: run_number = 0
#input_file_path = "./inputs/in_LF.json"
#### read input file ####
input = Inputs(input_file_path, run_number)

################################ set up grid and projection operators ################################################
grid = input.grid
filters = input.filters

##############  Time integration settings  ##############
#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#start, end time, time step
dt = input.dt_HF*day*10      #Note that we derive the low fidelity dt here, so the input file should still contain the high fidelity dt
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


#######################################################################################

    
###############################
# SPECIFY WHICH DATA TO STORE #
###############################


data_to_store = ['Q_LF', 'w_hat_n_LF']
if input.parametrization in ["TO","TO_track"]:
    data_to_store.append('tau')
    data_to_store.append('dQ')

#allocate memory
#framerate of storing data, plotting results (1 = every integration time step)
store_frame_rate = 1                                #store every time step
plot_frame_rate = np.floor(input.store_frame_rate*day/dt).astype('int') #plot every 10 days
 
#length of data array
S = np.floor(n_steps/store_frame_rate).astype('int')+1
S_fields = np.floor(n_steps/plot_frame_rate).astype('int')+2

samples = {}
if input.store_qoi_trajectories == True:
    store_fields = []
    store_scalars = []

    for q in range(len(data_to_store)):
        #assume a field contains the string '_hat_'
        if 'w_hat_n_LF' in data_to_store[q] or 'temp' in data_to_store[q]:
            store_fields.append(data_to_store[q])
            samples[data_to_store[q]] = np.zeros([S_fields, grid.N_LF_resolved, grid.N_LF_resolved]) + 0.0j
        #a scalar
        else:
            store_scalars.append(data_to_store[q])
            if data_to_store[q] == 'Q_LF':
                samples[data_to_store[q]] = np.zeros([S,input.N_Q_track+input.N_Q_save])
            else:
                samples[data_to_store[q]] = np.zeros([S,input.N_Q_track])
#######################################################################################

#######  forcing term   ########################
F_LF = 2**1.5*np.cos(5*grid.x_LF)*np.cos(5*grid.y_LF)
F_hat_LF = np.fft.fft2(F_LF)
if filters.use_gaussian_filter: F_hat_LF *= filters.gaussian_filter
##################################################

#########  inital conditions ######################
solution_state = Solution_state(input.time_integration_scheme, t, dt, nu, mu, grid, "LF")
solution_state.intialize_state(input, filters, grid, device)

####################################################

########### Load reference model ##################################################
# the reference model (HF model) is not executed at the same time, load a
# database containing the reference values for Q_i
if input.parametrization == "TO_track":  # offline tracking
    ref_reader = ReferenceFile_reader(input.parametrization_file_path, input.N_Q_track ,index_permutation=input.ref_file_permutation, device=device)

######### load base parametrization ########
if input.parametrization in ['CNN']:
    NN_parametrization = Plain_CNN_surrogate(file_path=input.parametrization_file_path, device=device)

######### load library for online tau prediction ######
if input.parametrization == "TO": 
    dQ_sampler = Dq_sampler(input.parametrization_file_path, device, independent_samples = input.sample_independent, sample_domain = input.sample_domain, use_MVG = input.use_MVG)


######### put everything on the GPU ######
# fields
F_hat_LF= torch.from_numpy(F_hat_LF).to(device=device)

# objects
filters.P_LF = torch.from_numpy(filters.P_LF).to(device=device)
if filters.use_gaussian_filter: filters.gaussian_filter = torch.from_numpy(filters.gaussian_filter).to(device=device)

grid.k_x_LF = torch.from_numpy(grid.k_x_LF).to(device=device)
grid.k_y_LF = torch.from_numpy(grid.k_y_LF).to(device=device)
grid.k_squared_nonzero_LF = torch.from_numpy(grid.k_squared_nonzero_LF).to(device=device)
grid.k_squared_LF = torch.from_numpy(grid.k_squared_LF).to(device=device)

filters.move_Pi_to_GPU(device)
to_mask = TO_masks(input.N_Q_track, device)

# update solution state with grid and filters on GPU
solution_state.set_grid_and_filter(grid, filters)
############################################################

print('*********************')
print('Solving forced dissipative vorticity equations on '+ str(device))
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')
print('*********************')

t0 = time.time()

############ time loop  ########################
#some counters for storing data
j = 0

# init some arrays
if input.parametrization in ["TO","TO_track"]:
    V_hat = torch.zeros([input.N_Q_track, grid.N_LF, grid.N_LF],dtype=torch.complex128,device=device)        
    T_i_hat = torch.zeros([input.N_Q_track, grid.N_LF, grid.N_LF],dtype=torch.complex128,device=device)
if input.parametrization in ["CNN"]:
    r_base = torch.zeros([grid.N_LF, grid.N_LF],dtype=torch.complex64,device=device)

# create sgs_functions which can be called within the RK4 loop to compute the SGS term
sgs_func = None
if input.discretization_SGS_term == 1:
    if input.parametrization == "smag":
        sgs_func = lambda w_hat_LF, psi_hat_LF: smag_parametrization(c_s = input.smag_constant, psi_hat_LF = psi_hat_LF, filters = filters, grid = grid)
    if input.parametrization == "CNN":
        sgs_func = lambda w_hat_LF, psi_hat_LF: \
            filters.fill_LF_with_resolved(LF_array = r_base, resolved_array = (torch.fft.fft2(
            NN_parametrization(w=torch.fft.ifft2(filters.filter_LF2resolved(w_hat_LF)).real.float(),
                            psi = torch.fft.ifft2(filters.filter_LF2resolved(psi_hat_LF)).real.float())
                            )).squeeze())
    

for n in range(n_steps):
    
    if np.mod(n, 50*int(day/dt)) == 0:
        print(f'day =  {n//int(day/dt)} of  {n_steps//int(day/dt)}')

    ########run the LF model#####################################################################
    #solve for next time step

    solution_state.time_step(F_hat_LF, sgs_func = sgs_func)
            

    ####################################
    #### SGS models  #######
    ####################################

    ####  CNN parametrization  #################################
    if (input.parametrization=="CNN") and (input.discretization_SGS_term ==2):

        r_temp = NN_parametrization(w=torch.fft.ifft2(filters.filter_LF2resolved(solution_state.w_hat_n)).real.float(),
                                    psi = torch.fft.ifft2(filters.filter_LF2resolved(solution_state.psi_hat_n)).real.float())
        r_base= filters.fill_LF_with_resolved(LF_array = r_base, resolved_array = (torch.fft.fft2(r_temp)).squeeze())

        ## Add sgs forcing to solution  ##
        solution_state.w_hat_np1 =  solution_state.w_hat_np1+solution_state.norm_factor*(r_base)
    #############################################################
    ##### SMAG  #################################################
    if input.parametrization=="smag" and (input.discretization_SGS_term ==2):
        r_base = smag_parametrization(c_s = input.smag_constant, psi_hat_LF = solution_state.psi_hat_n, filters = filters, grid = grid)
        ## Add sgs forcing to solution  ##
        solution_state.w_hat_np1 =  solution_state.w_hat_np1+solution_state.norm_factor*(r_base)
    ##############################################################
    
    #####  TO parametrization  ###################################
    if input.parametrization in ["TO","TO_track"]:
        
        if filters.use_gaussian_filter: 
            ## inverse gaussian filter ##
            solution_state.w_hat_n=solution_state.w_hat_n/filters.gaussian_filter
            solution_state.psi_hat_n=solution_state.psi_hat_n/filters.gaussian_filter
            solution_state.w_hat_np1=solution_state.w_hat_np1/filters.gaussian_filter
            solution_state.psi_hat_np1=solution_state.psi_hat_np1/filters.gaussian_filter

        ###### get dQ  ###############################
        if input.parametrization == "TO":
            dQ = dQ_sampler.sample()
        elif input.parametrization == "TO_track":
            ## we can use dQ at step n or at step n+1:
            # based on step n gives a relaxation type tracking, since we can't immediately correct for errors
            # based on step n+1 allows perfect tracking of the high fidelity QoIs, this can be seen as more aggressive modification
            if input.dQ_discretization == "n":
                Q_HF_n, Q_LF_n = get_QHF_QLF(input.targets, input.N_Q_track, solution_state.w_hat_n, solution_state.psi_hat_n, ref_reader= ref_reader, filters=filters, device=device )
                dQ = Q_HF_n-Q_LF_n
            elif input.dQ_discretization == "np1":
                if n==0: _ = ref_reader.next_QHF() # skip first QHF to get values at n+1
                Q_HF_np1, Q_LF_np1 = get_QHF_QLF(input.targets, input.N_Q_track, solution_state.w_hat_np1, solution_state.psi_hat_np1, ref_reader=ref_reader,filters=filters, device=device)
                dQ = Q_HF_np1-Q_LF_np1
            dQ = input.relax_constant*dQ
        
        ##############################################

        #QoI basis functions V
        for i in range(input.N_Q_track):
            if input.dQ_discretization in ["np1"]:
                V_hat[i] = filters.apply_P_i(evaluate_expression_simple(expression = input.V[i], psi_hat=solution_state.psi_hat_np1, w_hat=solution_state.w_hat_np1),i)     # V = dq_i/dw
                T_i_hat[i] = filters.apply_P_i(evaluate_expression_simple(expression = input.T_is[i],psi_hat=solution_state.psi_hat_np1, w_hat=solution_state.w_hat_np1),i) # basis functions of the orthogonal patterns
            if input.dQ_discretization == "n":
                V_hat[i] = filters.apply_P_i(evaluate_expression_simple(expression = input.V[i], psi_hat=solution_state.psi_hat_n, w_hat=solution_state.w_hat_n),i)
                T_i_hat[i] = filters.apply_P_i(evaluate_expression_simple(expression = input.T_is[i],psi_hat=solution_state.psi_hat_n, w_hat=solution_state.w_hat_n),i)
        
        #compute reduced eddy forcing
        r_tilde, _, _, _, tau = reduced_r_fast(V_hat, T_i_hat, dQ, masks = to_mask, device=device)

        ## Add sgs forcing to solution  ##
        if input.dQ_discretization == "n":
            EF_hat=  solution_state.norm_factor*r_tilde
        elif input.dQ_discretization == "np1":
            EF_hat= r_tilde
        solution_state.w_hat_np1 -=  EF_hat
        
        if filters.use_gaussian_filter: 
            ### apply Gaussian again  ############  
            solution_state.w_hat_n    =solution_state.w_hat_n*filters.gaussian_filter
            solution_state.psi_hat_n  =solution_state.psi_hat_n*filters.gaussian_filter
            solution_state.w_hat_np1  =solution_state.w_hat_np1*filters.gaussian_filter
            solution_state.psi_hat_np1=solution_state.psi_hat_np1*filters.gaussian_filter
    ######################################################################
    # update psi_hat_np1
    if (input.parametrization in ["TO","TO_track"]) or (input.parametrization in ["CNN","smag"] and input.discretization_SGS_term ==2):
        solution_state.psi_hat_np1 = get_psi_hat(solution_state.w_hat_np1, grid.k_squared_nonzero_LF)
    ###  Compute & store QoI to dictionary ###
    if input.store_qoi_trajectories == True:
        if filters.use_gaussian_filter:
            Q_LF = get_QLF(input.targets, input.N_Q_track+input.N_Q_save, w_hat_LF=solution_state.w_hat_n/filters.gaussian_filter, psi_hat_LF=solution_state.psi_hat_n/filters.gaussian_filter, filters=filters, device=device)
        else:
            Q_LF = get_QLF(input.targets,  input.N_Q_track+input.N_Q_save, w_hat_LF=solution_state.w_hat_n, psi_hat_LF=solution_state.psi_hat_n, filters=filters, device=device)
            
        #store the scalar QoI (every time step)
        for qoi in store_scalars:
            samples[qoi][n] = eval(qoi).cpu()
        # store fields (once every plot_frame_rate)
        if np.mod(n,plot_frame_rate) == 0:
            samples['w_hat_n_LF'][j] = filters.filter_LF2resolved(solution_state.w_hat_n).cpu()
            j += 1
    ###############################################
    
    #update variables
    solution_state.update_vars()
    
    ## check for nans ###########
    if torch.isnan(torch.sum(solution_state.w_hat_n)):
        print("NaN in solution. Interrupting time integration.")
        break
##### end of time loop  ##############################################################################################################
t1 = time.time()
print('Simulation time =', t1 - t0, 'seconds')

if input.parametrization == "TO_track": 
    ref_reader.close()


if input.store_qoi_trajectories == True:
    # store last fields
    samples['w_hat_n_LF'][j] = filters.filter_LF2resolved(solution_state.w_hat_n).cpu()
        
#store the samples
if input.store_qoi_trajectories == True:
    store_samples_hdf5(HOME,input.sim_ID,solution_state.t/day, data_to_store, samples)

# store state for restart
if input.store_final_state == True:
    restart_dic = {'w_hat_n_LF': filters.filter_LF2resolved(solution_state.w_hat_n).cpu().numpy(force=True),
                    'w_hat_nm1_LF': filters.filter_LF2resolved(solution_state.w_hat_nm1).cpu().numpy(force=True), 
                    'VgradW_hat_nm1_LF': filters.filter_LF2resolved(solution_state.VgradW_hat_nm1).cpu().numpy(force=True)}
    store_state(restart_dic, HOME, input.sim_ID, solution_state.t/day)

