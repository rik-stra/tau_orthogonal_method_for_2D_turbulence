"""
Author: Rik Hoekstra (18-06-2024)
"""

from typing import Any
import numpy as np
from scipy import stats
import torch
from math import ceil, floor



class Grids:
    def __init__(self, N_LF, N_HF, N_LF_resolved, N_HF_resolved):
        self.N_LF = N_LF; self.N_HF = N_HF
        self.N_LF_resolved = N_LF_resolved ; self.N_HF_resolved = N_HF_resolved
        self.dx_les = 2.0*np.pi/N_LF_resolved
        axis_LF = np.linspace(0, 2.0*np.pi, N_LF, endpoint=False)
        self.x_LF , self.y_LF = np.meshgrid(axis_LF, axis_LF)
        axis_HF = np.linspace(0, 2.0*np.pi, N_HF, endpoint=False)
        self.x_HF , self.y_HF = np.meshgrid(axis_HF, axis_HF)
        self.k_x_HF = 1j * np.fft.fftfreq(N_HF).reshape(-1,1)*N_HF
        self.k_y_HF = 1j * np.fft.fftfreq(N_HF).reshape(1,-1)*N_HF
        self.k_x_LF = 1j * np.fft.fftfreq(N_LF).reshape(-1,1)*N_LF
        self.k_y_LF = 1j * np.fft.fftfreq(N_LF).reshape(1,-1)*N_LF
        self.k_squared_HF, self.k_squared_nonzero_HF = _get_ksquared(self.k_x_HF, self.k_y_HF)
        self.k_squared_LF, self.k_squared_nonzero_LF = _get_ksquared(self.k_x_LF, self.k_y_LF)

def _get_ksquared(kx,ky):
    k_squared = kx**2 + ky**2
    k_squared_no_zero = np.copy(k_squared)
    k_squared_no_zero[0,0] = 1.0
    return k_squared, k_squared_no_zero

def get_dealiasing_filter(N,remove_alias,rfft = True, filter_size=None): # square filter!
    if filter_size == None: filter_size = N
    
    if rfft: k = np.fft.rfftfreq(filter_size).reshape(1,-1)*filter_size
    else: k = np.fft.fftfreq(filter_size).reshape(1,-1)*filter_size
    l = np.fft.fftfreq(filter_size).reshape(-1,1)*filter_size
    P = np.ones([l.shape[0], k.shape[1]],dtype=bool)
    if remove_alias:
        cutoff = int((N-1)/3)
        P = np.where(np.abs(k)<=cutoff,P,0)
        P = np.where(np.abs(l)<=cutoff,P,0)
    else:
        P = np.where(np.abs(k)<np.ceil(N/2),P,0)
        P = np.where(np.abs(l)<np.ceil(N/2),P,0)
        
    return P

#def get_P_k(k_min, k_max, N, binnumbers):   # round filter!!
#    P_k = np.zeros([N, N],dtype=bool)    
#    idx0, idx1 = np.where((binnumbers >= k_min) & (binnumbers <= k_max))
#    P_k[idx0, idx1] = 1.0
#    return P_k[0:N, 0:N]



class Round_filter:
    def __init__(self,k_min, k_max, filters):
        self.k_min = k_min
        self.k_max = k_max
        self.N_LF_resolved = filters.N_LF_resolved
        self.N_LF = filters.N_LF
        self.N_HF = filters.N_HF
        self.P_LF = torch.from_numpy(filters.P_LF)
        self.filter_LF = np.zeros([self.N_LF, self.N_LF])
        self.filter_HF = np.zeros([self.N_HF, self.N_HF])

        idx0, idx1 = np.where((filters.binnumbers_LF >= k_min) & (filters.binnumbers_LF <= k_max))
        self.filter_LF[idx0, idx1] = 1.0
        idx0, idx1 = np.where((filters.binnumbers_HF >= k_min) & (filters.binnumbers_HF <= k_max))
        self.filter_HF[idx0, idx1] = 1.0
        self.filter_HF *= filters.P_HF2LF

    def move_to(self,device):
        self.filter_LF = torch.from_numpy(self.filter_LF).to(device)
        self.filter_HF = torch.from_numpy(self.filter_HF).to(device)
        self.P_LF = self.P_LF.to(device)

    def __call__(self, x):
        num_batch_dims = len(x.shape[:-2])
        if x.shape[-2]==self.N_LF: return self.filter_LF.view([1]*num_batch_dims+[x.shape[-2],x.shape[-1]])*x
        elif x.shape[-2]==self.N_HF: return self.filter_HF*x
        elif x.shape[-2]==self.N_LF_resolved: return self.filter_LF[self.P_LF==1].reshape([1]*num_batch_dims+[self.N_LF_resolved,self.N_LF_resolved])*x
        else: print("No filter for this input dimensions:" , x.shape)


class Filters:
    def __init__(self,grid:Grids, remove_alias:bool, use_gaussian_filter:bool):

        self.N_LF = grid.N_LF
        self.N_HF = grid.N_HF
        self.N_LF_resolved = grid.N_LF_resolved
        self.N_HF_resolved = grid.N_HF_resolved
        self.remove_alias = remove_alias
        
        self.Ncutoff_HF=int(grid.N_HF/(2+remove_alias))
        self.Ncutoff_LF=int(grid.N_LF/(2+remove_alias))
        self.P_HF = get_dealiasing_filter(grid.N_HF,remove_alias,rfft=False)
        self.P_LF = get_dealiasing_filter(grid.N_LF,remove_alias,rfft=False)
        self.P_HF2LF = get_dealiasing_filter(grid.N_LF,remove_alias,rfft=False,filter_size=grid.N_HF)
        self.use_gaussian_filter = use_gaussian_filter
        if use_gaussian_filter:
            self.gaussian_filter = np.exp(grid.k_squared_LF*(2*grid.dx_les)**2/24)
        self.binnumbers_LF, self.bins_LF = freq_map(grid.N_LF, self.Ncutoff_LF, grid.k_x_LF, grid.k_y_LF)
        self.N_bins_LF = self.bins_LF.size
        self.binnumbers_HF, self.bins_HF = freq_map(grid.N_HF, self.Ncutoff_HF, grid.k_x_HF, grid.k_y_HF)
        self.N_bins_HF = self.bins_HF.size

        # round filters for Qoi
        self.P_i_unique = []  # contains each filter only once
        self.P_i = []         # contains index of filter in P_i_unique for each Qoi

    def apply_P_i(self, x, i):
        return self.P_i_unique[self.P_i[i]](x)
        
    def initialize_P_i(self, kmin, kmax):
        unique = -1
        for j in range(len(self.P_i_unique)):
            if self.P_i_unique[j].k_min == kmin and self.P_i_unique[j].k_max == kmax:
                unique = j
        if unique==-1: 
            self.P_i_unique.append(Round_filter(kmin,kmax,self))
            self.P_i.append(len(self.P_i_unique)-1)
        else:
            self.P_i.append(unique)

    def move_Pi_to_GPU(self,device):
        for p in self.P_i_unique:
            p.move_to(device)

    def filter_HF2LF(self, u_HF: np.ndarray):
        """creates low fidelity field from high fidelity field 

        Args:
            u_HF (_type_): the high fidelity field
            c (_type_): 
            

        Returns:
            low fidelity field: N_LF x N_LF
        """
        res = np.zeros((self.N_LF,self.N_LF),dtype=u_HF.dtype)
        res[self.P_LF==1] = u_HF[self.P_HF2LF==1]*(self.N_LF/self.N_HF)**2
        if self.use_gaussian_filter:
            res *= self.gaussian_filter
        return res
    
    def filter_HF2LF_torch(self, u_HF):
        """creates low fidelity field from high fidelity field 

        Args:
            u_HF (_type_): the high fidelity field
            c (_type_): 
            

        Returns:
            low fidelity field: N_LF x N_LF
        """
        res = torch.zeros((self.N_LF,self.N_LF),dtype=u_HF.dtype,device=u_HF.device)
        res[self.P_LF==1] = u_HF[self.P_HF2LF==1]*(self.N_LF/self.N_HF)**2
        if self.use_gaussian_filter:
            res *= self.gaussian_filter
        return res
    
    def filter_LF2resolved(self, x):
        if not self.remove_alias:
            return x
        else:
            return x[self.P_LF==1].reshape((self.N_LF_resolved,self.N_LF_resolved))*(self.N_LF_resolved/self.N_LF)**2
    
    def filter_HF2resolved(self, x):
        if not self.remove_alias:
            return x
        else:
            return x[self.P_HF==1].reshape((self.N_HF_resolved,self.N_HF_resolved))*(self.N_HF_resolved/self.N_HF)**2
    
    def fill_LF_with_resolved(self, LF_array, resolved_array):
        if not self.remove_alias:
            LF_array = resolved_array
        else:
            a = ceil(self.N_LF_resolved/2)
            b = -floor(self.N_LF_resolved/2)
            LF_array[:a,:a] = resolved_array[:a,:a]
            LF_array[:a,b:] = resolved_array[:a,b:]
            LF_array[b:,:a] = resolved_array[b:,:a]
            LF_array[b:,b:] = resolved_array[b:,b:]
        return LF_array*(self.N_LF/self.N_LF_resolved)**2

#########################
## SPECTRUM SUBROUTINES #
#########################

def freq_map(N, N_cutoff, kx, ky):
    """
    Map 2D frequencies to a 1D bin (kx, ky) --> k
    where k = 0, 1, ..., sqrt(2)*Ncutoff
    """
   
    #edges of 1D wavenumber bins
    bins = np.arange(-0.5, np.ceil(2**0.5*N_cutoff)+1)
    #fmap = np.zeros([N,N]).astype('int')
    

    #Euclidian distance of frequencies kx and ky
    dist = np.abs(np.sqrt(kx**2 + ky**2))
                
    #find 1D bin index of dist
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    
    binnumbers -= 1
            
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat,psi_hat, P, N, N_bins, binnumbers):
  
    
    E_hat = np.real(-0.5*psi_hat*np.conjugate(w_hat)/N**4)
    Z_hat = np.real(0.5*w_hat*np.conjugate(w_hat)/N**4)
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    E_spec[:np.max(binnumbers)+1]=np.bincount(binnumbers.flatten(), weights=E_hat.flatten())
    Z_spec[:np.max(binnumbers)+1]=np.bincount(binnumbers.flatten(), weights=Z_hat.flatten())
    return E_spec, Z_spec

#############################
#  END SPECTRUM SUBROUTINES #
#############################