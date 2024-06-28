import numpy as np
import h5py
import torch
from aux_code.filters import Filters

# ReferenceFile_reader, TO_masks, get_w_hat_np1, get_psi_hat, get_QHF_QLF, evaluate_expression_simple, reduced_r_fast, get_QLF
######################
# SOLVER SUBROUTINES #
######################

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n using the pseudo-spectral method

    Args:
        w_hat_n (complex): fourier coef of vorticity
        P (_type_): filter
        kx (_type_): first entry wavenumer vector
        ky (_type_): second entry wavenumber vector
        k_squared_no_zero (_type_): 

    Returns:
        _type_: filtered fourier coefficients for VgradW_hat
    """
    u_n = torch.fft.ifft2(-ky*psi_hat_n).real
    v_n = torch.fft.ifft2(kx*psi_hat_n).real
    #compute jacobian in physical space Non-conservative
    w_x_n = torch.fft.ifft2(kx*w_hat_n).real
    w_y_n = torch.fft.ifft2(ky*w_hat_n).real
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    #return to spectral space
    VgradW_hat_n_nc = torch.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    #compute jacobian conservative
    w_n = torch.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*torch.fft.fft2(u_n*w_n)+ky* torch.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P

    VgradW_hat_n = 0.5* (VgradW_hat_n_c+VgradW_hat_n_nc)

    return VgradW_hat_n

#pseudo-spectral technique to solve for Fourier coefs of Jacobian using Numpy 
def compute_VgradW_hat_np(w_hat_n, psi_hat_n, P, kx, ky):
    """Computes u_n*w_x_n + v_n*w_y_n via NumPy using the pseudo-spectral method

    Args:
        w_hat_n (complex): fourier coef of vorticity
        P (_type_): filter
        kx (_type_): first entry wavenumer vector
        ky (_type_): second entry wavenumber vector
        k_squared_no_zero (_type_): 

    Returns:
        _type_: filtered fourier coefficients for VgradW_hat
    """
    u_n = np.fft.ifft2(-ky*psi_hat_n).real
    v_n = np.fft.ifft2(kx*psi_hat_n).real
    #compute jacobian in physical space Non-conservative
    w_x_n = np.fft.ifft2(kx*w_hat_n).real
    w_y_n = np.fft.ifft2(ky*w_hat_n).real
    VgradW_n_nc = u_n*w_x_n + v_n*w_y_n
    #return to spectral space
    VgradW_hat_n_nc = np.fft.fft2(VgradW_n_nc)
    VgradW_hat_n_nc *= P
    
    #compute jacobian conservative
    w_n = np.fft.ifft2(w_hat_n).real
    VgradW_hat_n_c = kx*np.fft.fft2(u_n*w_n)+ky* np.fft.fft2(v_n*w_n)
    VgradW_hat_n_c *= P

    VgradW_hat_n = 0.5* (VgradW_hat_n_c+VgradW_hat_n_nc)

    return VgradW_hat_n

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(method,dt,mu,nu,w_hat_n, psi_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, kx, ky, k_squared, F_hat, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, psi_hat_n, P, kx, ky)
    
    if method == "AB/BDI2":
        #solve for next time step according to AB/BDI2 scheme
        w_hat_np1 = norm_factor*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                                2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    elif method == "AB/CN":
        #solve for next time step according to AB/CN scheme
        w_hat_np1 = norm_factor*P*(2/dt*w_hat_n + nu*k_squared*w_hat_n - (3*VgradW_hat_n - VgradW_hat_nm1) + mu*(2*F_hat-w_hat_n)- sgs_hat)
    
    return w_hat_np1, VgradW_hat_n


#return the fourier coefs of the stream function
def get_psi_hat(w_hat_n, k_squared_no_zero):

    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0

    return psi_hat_n


############################
# get QoI
############################

class ReferenceFile_reader:
    def __init__(self, file_name, N_Q, index_permutation = 'id', device='cpu'):
        self.ref_file = h5py.File(file_name, 'r')
        self.size_file_ref = self.ref_file['Q_HF'].shape[0]
        self.index_permutation = index_permutation
        self.cache_size = 10000
        self.skip = 1 # number of HF time steps saved per LF time step
        self.start_index = 0

        if isinstance(index_permutation,list):
            self.permute = np.array(index_permutation,dtype=int)
        elif isinstance(index_permutation,str):
            self.permute  = np.arange(N_Q,dtype=int)
        else:
            print("Please provide index permutation as a list")

        self.cache = torch.zeros(self.cache_size,N_Q,device=device)
        self.read_new_vals()

    def next_QHF(self):
        if self.cache_index >= self.cache_size:
            self.read_new_vals()
        self.cache_index += 1
        return self.cache[self.cache_index-1, :]

    def read_new_vals(self):
        if self.start_index*self.skip >= self.size_file_ref:
            print("End of reference file reached")
        else:
            self.cache[:,:] = torch.from_numpy(self.ref_file['Q_HF'][self.start_index*self.skip : (self.start_index+self.cache_size)*self.skip : self.skip, self.permute]).to(device=self.cache.device)
            self.start_index += self.cache_size
            self.cache_index = 0


    def close(self):
        self.ref_file.close()

def get_QHF_QLF(targets, N_Q, w_hat_LF, psi_hat_LF, ref_reader = None, filters=None, device='cpu' ):
    """Computes the quatities of interest in the high fidelity solution and in the low fidelity solution. 
    When QHF is read from a file one can specify an index_permutation if the quatities of interest are not in the same order in the reference file: 
    use index_permutation = [1,4] if the needed quatities are in column 2 and 5 for example.
    """

    # read Q_HF
    Q_HF = ref_reader.next_QHF()
    
    # computes Q_LF
    Q_LF = get_QLF(targets, N_Q, w_hat_LF=w_hat_LF, psi_hat_LF=psi_hat_LF, filters=filters, device=device)
    return Q_HF, Q_LF

def get_QLF(targets, N_Q, w_hat_LF, psi_hat_LF, filters:Filters, device):
    """Computes the quatities of interest in the low fidelity solution
    """

    N_LF = np.shape(w_hat_LF)[0]
    Q_LF = torch.zeros(N_Q,device=device)
    for i in range(N_Q):
        Q_LF[i] = get_qoi(filters.apply_P_i(w_hat_LF,i), filters.apply_P_i(psi_hat_LF,i),
                              targets[i], N_LF).real
    
    return Q_LF

def get_qoi(w_hat_n, psi_hat_n, target, N):

    """
    compute the Quantity of Interest defined by the string target
    """

    #energy (-psi, omega)/2
    if target == 'e':
        return 0.5*torch.dot(-psi_hat_n.flatten(), torch.conj(w_hat_n.flatten()))/N**4
    
    #enstrophy (omega, omega)/2
    elif target == 'z':
        return 0.5*torch.dot(w_hat_n.flatten(), torch.conj(w_hat_n.flatten()))/N**4
    
    #squared streamfunction (psi, psi)/2
    elif target == 's':
        return 0.5*torch.dot(psi_hat_n.flatten(), torch.conj(psi_hat_n.flatten()))/N**4

    else:
        print(target, 'IS AN UNKNOWN QUANTITY OF INTEREST')
        import sys; sys.exit()

def inner_products(V_hat, T_i, N):

    """
    Compute all the inner products (V_i, T_{i})
    """
    N_squared = N**2
    V_hat_flat = V_hat.reshape(list(V_hat.shape[:-2])+[N_squared])
    T_i_flat = T_i.reshape(list(T_i.shape[:-2])+ [N_squared])
    return torch.matmul(V_hat_flat, torch.conj(T_i_flat).transpose(-2,-1))/N_squared**2

def evaluate_expression_simple(expression, psi_hat, w_hat):
        return eval(expression)



