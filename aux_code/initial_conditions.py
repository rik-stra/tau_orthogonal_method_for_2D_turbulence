"""
File containing functions to get the initial conditions for the simulation.
Author: Rik Hoekstra (18-06-2024)"""

import numpy as np
from aux_code.filters import Filters, Grids
from aux_code.functions_for_solver import compute_VgradW_hat_np, get_psi_hat
import h5py

def get_initial_conditions(from_restart_file:bool, filters:Filters, grid:Grids, fname:str="none", filter_restart_state=False):
    """
    Get the initial conditions for the simulation.

    Args:
        from_restart_file (bool): Flag indicating whether to load initial conditions from a restart file.
        filters (Filters): Filters object containing filter information.
        grid (Grids): Grids object containing grid information.
        fname (str, optional): File name of the restart file. Defaults to "none".
        filter_restart_state (bool, optional): Flag indicating whether to filter the restart state. Defaults to False.

    Returns:
        tuple: Tuple containing the initial conditions for the simulation.
    """
    if from_restart_file:
        return initial_conditions_from_restart(fname, filters, grid, filter_restart_state)
    else:
        return create_initial_conditions(grid, filters)

def initial_conditions_from_restart(fname, filters:Filters, grid:Grids, filter_restart_state=False):
    """
    Get the initial conditions from a restart file.

    Args:
        fname (str): File name of the restart file.
        filters (Filters): Filters object containing filter information.
        grid (Grids): Grids object containing grid information.
        filter_restart_state (bool, optional): Flag indicating whether to filter the restart state. Defaults to False.

    Returns:
        tuple: Tuple containing the initial conditions for the simulation.
    """
    # define high fidelity arrays
    VgradW_hat_nm1_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=np.complex128)
    VgradW_hat_nm10_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=np.complex128)
    w_hat_n_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=np.complex128)
    w_hat_nm1_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=np.complex128)
    w_hat_nm10_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=np.complex128)

    # create HDF5 file
    h5f = h5py.File(fname, 'r')

    if "w_hat_n_LF" in h5f.keys():
        w_hat_n_LF = np.zeros((grid.N_LF,grid.N_LF),dtype=np.complex128)
        w_hat_nm1_LF = np.zeros((grid.N_LF,grid.N_LF),dtype=np.complex128)
        VgradW_hat_nm1_LF = np.zeros((grid.N_LF,grid.N_LF),dtype=np.complex128)

        w_hat_n_LF[filters.P_LF==1]= np.array(h5f['w_hat_n_LF']).flatten()*(filters.N_LF/filters.N_LF_resolved)**2
        w_hat_nm1_LF[filters.P_LF==1]= np.array(h5f['w_hat_nm1_LF']).flatten()*(filters.N_LF/filters.N_LF_resolved)**2
        VgradW_hat_nm1_LF[filters.P_LF==1]= np.array(h5f['VgradW_hat_nm1_LF']).flatten()*(filters.N_LF/filters.N_LF_resolved)**2
        
        psi_hat_nm1_LF = get_psi_hat(w_hat_nm1_LF, grid.k_squared_nonzero_LF) 
        psi_hat_n_LF = get_psi_hat(w_hat_n_LF, grid.k_squared_nonzero_LF)

    else: # restart from highfidelity
        VgradW_hat_nm1_HF[filters.P_HF==1]= np.array(h5f['VgradW_hat_nm1_HF']).flatten()*(filters.N_HF/filters.N_HF_resolved)**2
        VgradW_hat_nm10_HF[filters.P_HF==1]= np.array(h5f['VgradW_hat_nm10_HF']).flatten()*(filters.N_HF/filters.N_HF_resolved)**2
        
        w_hat_n_HF[filters.P_HF==1]= np.array(h5f['w_hat_n_HF']).flatten()*(filters.N_HF/filters.N_HF_resolved)**2
        
        w_hat_nm1_HF[filters.P_HF==1]= np.array(h5f['w_hat_nm1_HF']).flatten()*(filters.N_HF/filters.N_HF_resolved)**2
        w_hat_nm10_HF[filters.P_HF==1]= np.array(h5f['w_hat_nm10_HF']).flatten()*(filters.N_HF/filters.N_HF_resolved)**2
        h5f.close()

        if (filter_restart_state): w_hat_n_HF *= filters.P_HF2LF; w_hat_nm1_HF *= filters.P_HF2LF; VgradW_hat_nm1_HF *= filters.P_HF2LF

        # determine low fidelity states
        w_hat_nm1_LF = filters.filter_HF2LF(w_hat_nm10_HF)
        psi_hat_nm1_LF = get_psi_hat(w_hat_nm1_LF, grid.k_squared_nonzero_LF)  
        VgradW_hat_nm1_LF = filters.filter_HF2LF(VgradW_hat_nm10_HF)

        w_hat_n_LF = filters.filter_HF2LF(w_hat_n_HF)
        psi_hat_n_LF = get_psi_hat(w_hat_n_LF, grid.k_squared_nonzero_LF)
        
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF, grid.k_squared_nonzero_HF)
    psi_hat_nm1_HF = get_psi_hat(w_hat_nm1_HF, grid.k_squared_nonzero_HF)

    return w_hat_n_HF, w_hat_n_LF, w_hat_nm1_HF, w_hat_nm1_LF, VgradW_hat_nm1_HF, VgradW_hat_nm1_LF, psi_hat_nm1_LF, psi_hat_n_LF, psi_hat_n_HF, psi_hat_nm1_HF

def create_initial_conditions(grid:Grids, filters:Filters):
    """
    Create the initial conditions for the simulation.

    Args:
        grid (Grids): Grids object containing grid information.
        filters (Filters): Filters object containing filter information.

    Returns:
        tuple: Tuple containing the initial conditions for the simulation.
    """
    # initial condition
    w_LF = np.sin(4.0*grid.x_LF)*np.sin(4.0*grid.y_LF) + 0.4*np.cos(3.0*grid.x_LF)*np.cos(3.0*grid.y_LF) + \
           0.3*np.cos(5.0*grid.x_LF)*np.cos(5.0*grid.y_LF) + 0.02*np.sin(grid.x_LF) + 0.02*np.cos(grid.y_LF)

    w_hat_n_LF = filters.P_LF*np.fft.fft2(w_LF)
    w_hat_n_HF = np.zeros((grid.N_HF,grid.N_HF),dtype=complex)
    w_hat_n_HF[filters.P_HF2LF==1] = w_hat_n_LF[filters.P_LF==1]*(grid.N_HF/grid.N_LF)**2

    psi_hat_n_LF = get_psi_hat(w_hat_n_LF, grid.k_squared_nonzero_LF)
    psi_hat_n_HF = get_psi_hat(w_hat_n_HF, grid.k_squared_nonzero_HF)

    # initial Fourier coefficients at time n-1
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    
    # initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat_np(w_hat_n_HF, psi_hat_n_HF, filters.P_HF, grid.k_x_HF, grid.k_y_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    VgradW_hat_n_LF = compute_VgradW_hat_np(w_hat_n_LF, psi_hat_n_LF, filters.P_LF, grid.k_x_LF, grid.k_y_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

    psi_hat_nm1_HF = get_psi_hat(w_hat_nm1_HF, grid.k_squared_nonzero_HF)
    psi_hat_nm1_LF = get_psi_hat(w_hat_nm1_LF, grid.k_squared_nonzero_LF)  
    return w_hat_n_HF, w_hat_n_LF, w_hat_nm1_HF, w_hat_nm1_LF, VgradW_hat_nm1_HF, VgradW_hat_nm1_LF, psi_hat_nm1_LF, psi_hat_n_LF, psi_hat_n_HF, psi_hat_nm1_HF
