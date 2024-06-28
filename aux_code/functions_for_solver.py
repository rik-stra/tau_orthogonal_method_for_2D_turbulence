"""
This file contains functions to compute
- the non linear term in the vorticity equation
- the stream function from the vorticity
- the qoi 
- the inner products of two fields

Author: Rik Hoekstra (18-06-2024)
"""

import numpy as np
import h5py
import torch
from aux_code.filters import Filters


######################
# SOLVER SUBROUTINES #
######################

def compute_VgradW_hat(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n using the pseudo-spectral method

    Args:
        w_hat_n (complex): Fourier coefficients of vorticity
        psi_hat_n (complex): Fourier coefficients of stream function
        P (array): dealiassing filter
        kx (array): First entry wavenumber vector
        ky (array): Second entry wavenumber vector

    Returns:
        complex: Filtered Fourier coefficients for VgradW_hat
    """
    # Compute velocity components in physical space
    u_n = torch.fft.ifft2(-ky*psi_hat_n).real
    v_n = torch.fft.ifft2(kx*psi_hat_n).real
    
    # Compute vorticity gradients in physical space
    w_x_n = torch.fft.ifft2(kx*w_hat_n).real
    w_y_n = torch.fft.ifft2(ky*w_hat_n).real
    
    # Compute VgradW_n_nc (non-conservative) in physical space
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    
    # Convert VgradW_n_nc to spectral space and apply filter
    VgradW_hat_n_nc = torch.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    # Compute VgradW_n_c (conservative) in spectral space and apply filter
    w_n = torch.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*torch.fft.fft2(u_n*w_n) + ky*torch.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P
    
    # Compute VgradW_n as the average of VgradW_n_c and VgradW_n_nc
    VgradW_hat_n = 0.5 * (VgradW_hat_n_c + VgradW_hat_n_nc)

    return VgradW_hat_n

def compute_VgradW_hat_np(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n using the pseudo-spectral method with NumPy

    Args:
        w_hat_n (complex): Fourier coefficient of vorticity
        psi_hat_n (complex): Fourier coefficient of stream function
        P (filter): Filter object
        kx (array): First entry wavenumber vector
        ky (array): Second entry wavenumber vector

    Returns:
        complex: Filtered Fourier coefficients for VgradW_hat
    """
    # Compute velocity components in physical space
    u_n = np.fft.ifft2(-ky*psi_hat_n).real
    v_n = np.fft.ifft2(kx*psi_hat_n).real
    
    # Compute vorticity gradients in physical space
    w_x_n = np.fft.ifft2(kx*w_hat_n).real
    w_y_n = np.fft.ifft2(ky*w_hat_n).real
    
    # Compute VgradW_n_nc (non-conservative) in physical space
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    
    # Convert VgradW_n_nc to spectral space and apply filter
    VgradW_hat_n_nc = np.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    # Compute VgradW_n_c (conservative) in spectral space and apply filter
    w_n = np.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*np.fft.fft2(u_n*w_n) + ky*np.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P
    
    # Compute VgradW_n as the average of VgradW_n_c and VgradW_n_nc
    VgradW_hat_n = 0.5 * (VgradW_hat_n_c + VgradW_hat_n_nc)

    return VgradW_hat_n


def get_psi_hat(w_hat_n, k_squared_no_zero):
    """Computes the Fourier coefficients of the stream function

    Args:
        w_hat_n (complex): Fourier coefficients of vorticity
        k_squared_no_zero (array): Squared wavenumber vector excluding zero

    Returns:
        complex: Fourier coefficients of stream function
    """
    psi_hat_n = w_hat_n / k_squared_no_zero
    psi_hat_n[0, 0] = 0.0

    return psi_hat_n

############################
# get QoI
############################

class ReferenceFile_reader:
    def __init__(self, file_name, N_Q, index_permutation='id', device='cpu'):
        """Class for reading reference file with ref trajectories for QoI.

        Args:
            file_name (str): Path to the reference file
            N_Q (int): Number of quantities of interest
            index_permutation (list or str, optional): Index permutation for quantities of interest. Defaults to 'id'.
            device (str, optional): Device to store the cache. Defaults to 'cpu'.
        """
        self.ref_file = h5py.File(file_name, 'r')
        self.size_file_ref = self.ref_file['Q_HF'].shape[0]
        self.index_permutation = index_permutation
        self.cache_size = 10000
        self.skip = 1  # Number of HF time steps saved per LF time step
        self.start_index = 0

        if isinstance(index_permutation, list):
            self.permute = np.array(index_permutation, dtype=int)
        elif isinstance(index_permutation, str):
            self.permute = np.arange(N_Q, dtype=int)
        else:
            print("Please provide index permutation as a list")

        self.cache = torch.zeros(self.cache_size, N_Q, device=device)
        self.read_new_vals()

    def next_QHF(self):
        """Get the next set of quantities of interest in the reference file

        Returns:
            tensor: Quantities of interest in the high fidelity solution
        """
        if self.cache_index >= self.cache_size:
            self.read_new_vals()
        self.cache_index += 1
        return self.cache[self.cache_index-1, :]

    def read_new_vals(self):
        """Read new values from the reference file and update the cache"""
        if self.start_index * self.skip >= self.size_file_ref:
            print("End of reference file reached")
        else:
            self.cache[:, :] = torch.from_numpy(self.ref_file['Q_HF'][self.start_index*self.skip : (self.start_index+self.cache_size)*self.skip : self.skip, self.permute]).to(device=self.cache.device)
            self.start_index += self.cache_size
            self.cache_index = 0

    def close(self):
        """Close the reference file"""
        self.ref_file.close()

def get_QHF_QLF(targets, N_Q, w_hat_LF, psi_hat_LF, ref_reader=None, filters=None, device='cpu'):
    """Gets the quantities of interest in the high fidelity solution and computes them for the low fidelity solution

    Args:
        targets (list): List of target quantities of interest
        N_Q (int): Number of quantities of interest
        w_hat_LF (complex): Fourier coefficients of vorticity in the low fidelity solution
        psi_hat_LF (complex): Fourier coefficients of stream function in the low fidelity solution
        ref_reader (ReferenceFile_reader, optional): Reference file reader object. Defaults to None.
        filters (Filters, optional): Filter object. Defaults to None.
        device (str, optional): Device to store the tensors. Defaults to 'cpu'.

    Returns:
        tensor: Quantities of interest in the high fidelity solution
        tensor: Quantities of interest in the low fidelity solution
    """
    # Read Q_HF
    Q_HF = ref_reader.next_QHF()
    
    # Compute Q_LF
    Q_LF = get_QLF(targets, N_Q, w_hat_LF=w_hat_LF, psi_hat_LF=psi_hat_LF, filters=filters, device=device)
    
    return Q_HF, Q_LF

def get_QLF(targets, N_Q, w_hat_LF, psi_hat_LF, filters: Filters, device):
    """Computes the quantities of interest in the low fidelity solution

    Args:
        targets (list): List of target quantities of interest
        N_Q (int): Number of quantities of interest
        w_hat_LF (complex): Fourier coefficients of vorticity in the low fidelity solution
        psi_hat_LF (complex): Fourier coefficients of stream function in the low fidelity solution
        filters (Filters): Filter object
        device (str): Device to store the tensors

    Returns:
        tensor: Quantities of interest in the low fidelity solution
    """
    N_LF = np.shape(w_hat_LF)[0]
    Q_LF = torch.zeros(N_Q, device=device)
    for i in range(N_Q):
        Q_LF[i] = get_qoi(filters.apply_P_i(w_hat_LF, i), filters.apply_P_i(psi_hat_LF, i), targets[i], N_LF).real
    
    return Q_LF

def get_qoi(w_hat_n, psi_hat_n, target, N):
    """Compute the Quantity of Interest defined by the target string

    Args:
        w_hat_n (complex): Fourier coefficients of vorticity
        psi_hat_n (complex): Fourier coefficients of stream function
        target (str): Target quantity of interest
        N (int): Size of the domain

    Returns:
        float: Quantity of interest value
    """
    # Energy (-psi, omega)/2
    if target == 'e':
        return 0.5 * torch.dot(-psi_hat_n.flatten(), torch.conj(w_hat_n.flatten())) / N**4
    
    # Enstrophy (omega, omega)/2
    elif target == 'z':
        return 0.5 * torch.dot(w_hat_n.flatten(), torch.conj(w_hat_n.flatten())) / N**4
    
    # Squared streamfunction (psi, psi)/2
    elif target == 's':
        return 0.5 * torch.dot(psi_hat_n.flatten(), torch.conj(psi_hat_n.flatten())) / N**4

    else:
        print(target, 'IS AN UNKNOWN QUANTITY OF INTEREST')
        import sys; sys.exit()

def inner_products(V_hat, T_i, N):
    """Compute all the inner products (V_i, T_{i})

    Args:
        V_hat (complex): Fourier coefficients of V_i
        T_i (complex): Fourier coefficients of T_i
        N (int): Size of the domain

    Returns:
        complex: Inner products (V_i, T_{i})
    """
    N_squared = N**2
    V_hat_flat = V_hat.reshape(list(V_hat.shape[:-2]) + [N_squared])
    T_i_flat = T_i.reshape(list(T_i.shape[:-2]) + [N_squared])
    return torch.matmul(V_hat_flat, torch.conj(T_i_flat).transpose(-2, -1)) / N_squared**2

def evaluate_expression_simple(expression, psi_hat, w_hat):
    """Evaluate a simple expression using the stream function and vorticity Fourier coefficients

    Args:
        expression (str): Expression to evaluate
        psi_hat (complex): Fourier coefficients of stream function
        w_hat (complex): Fourier coefficients of vorticity

    Returns:
        Any: Result of the evaluated expression
    """
    return eval(expression)



