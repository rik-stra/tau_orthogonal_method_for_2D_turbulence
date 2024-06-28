"""
This file contains the implementation of grids and various filters used in the tau-orthogonal method for 2D turbulence simulation.

The file includes the following classes and functions:
- Grids: A class that represents grid information for low and high fidelity fields.
- _get_ksquared: A function that calculates the squared wavenumber.
- get_dealiasing_filter: A function that returns the dealiasing filter.
- Round_filter: A class that represents a round filter for a specific range of wavenumbers.
- Filters: A class that represents a collection of filters used in the simulation.
- freq_map: A function that maps 2D frequencies to a 1D bin.

Author: Rik Hoekstra (18-06-2024)
"""

from typing import Any
import numpy as np
from scipy import stats
import torch
from math import ceil, floor

class Grids:
    def __init__(self, N_LF, N_HF, N_LF_resolved, N_HF_resolved):
        """
        Initialize the Grids class. Contains grid information for low and high fidelity fields.

        Args:
            N_LF (int): Number of points in the low fidelity grid.
            N_HF (int): Number of points in the high fidelity grid.
            N_LF_resolved (int): Number of points in the low fidelity resolved grid.
            N_HF_resolved (int): Number of points in the high fidelity resolved grid.
        """
        self.N_LF = N_LF
        self.N_HF = N_HF
        self.N_LF_resolved = N_LF_resolved
        self.N_HF_resolved = N_HF_resolved
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
    """
    Calculate the squared wavenumber.

    Args:
        kx (ndarray): x-component of the wavenumber.
        ky (ndarray): y-component of the wavenumber.

    Returns:
        ndarray: Squared wavenumber.
        ndarray: Squared wavenumber with zero element replaced by 1.0.
    """
    k_squared = kx**2 + ky**2
    k_squared_no_zero = np.copy(k_squared)
    k_squared_no_zero[0,0] = 1.0
    return k_squared, k_squared_no_zero

def get_dealiasing_filter(N,remove_alias,rfft = True, filter_size=None): # square filter!
    """
    Get the dealiasing filter.

    Args:
        N (int): Number of points in the grid.
        remove_alias (bool): Flag to remove aliasing.
        rfft (bool, optional): Flag to use real FFT. Defaults to True.
        filter_size (int, optional): Size of the filter. Defaults to None.

    Returns:
        ndarray: Dealiasing filter.
    """
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

class Round_filter:
    def __init__(self,k_min, k_max, filters):
        """
        Initialize the Round_filter class.

        Args:
            k_min (float): Minimum wavenumber.
            k_max (float): Maximum wavenumber.
            filters (Filters): Filters object.
        """
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
        """
        Move the filter to the specified device.

        Args:
            device (str): Device to move the filter to.
        """
        self.filter_LF = torch.from_numpy(self.filter_LF).to(device)
        self.filter_HF = torch.from_numpy(self.filter_HF).to(device)
        self.P_LF = self.P_LF.to(device)

    def __call__(self, x):
        """
        Apply the filter to the input.

        Args:
            x (ndarray): Input array.

        Returns:
            ndarray: Filtered array.
        """
        num_batch_dims = len(x.shape[:-2])
        if x.shape[-2]==self.N_LF: return self.filter_LF.view([1]*num_batch_dims+[x.shape[-2],x.shape[-1]])*x
        elif x.shape[-2]==self.N_HF: return self.filter_HF*x
        elif x.shape[-2]==self.N_LF_resolved: return self.filter_LF[self.P_LF==1].reshape([1]*num_batch_dims+[self.N_LF_resolved,self.N_LF_resolved])*x
        else: print("No filter for this input dimensions:" , x.shape)

class Filters:
    def __init__(self,grid:Grids, remove_alias:bool, use_gaussian_filter:bool):
        """
        Initialize the Filters class.

        Args:
            grid (Grids): Grids object.
            remove_alias (bool): Flag to remove aliassing error.
            use_gaussian_filter (bool): Flag to use Gaussian filter.
        """
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
        """
        Apply the filter P_i to the input.

        Args:
            x (ndarray): Input array.
            i (int): Index of the filter.

        Returns:
            ndarray: Filtered array.
        """
        return self.P_i_unique[self.P_i[i]](x)
        
    def initialize_P_i(self, kmin, kmax):
        """
        Initialize the P_i filter. If the filter already exists, the index is appended to P_i. Otherwise, a new filter is created.

        Args:
            kmin (float): Minimum wavenumber.
            kmax (float): Maximum wavenumber.
        """
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
        """
        Move the P_i filters to the specified device.

        Args:
            device (str): Device to move the filters to.
        """
        for p in self.P_i_unique:
            p.move_to(device)

    def filter_HF2LF(self, u_HF: np.ndarray):
        """
        Create low fidelity field from high fidelity field. For numpy arrays.

        Args:
            u_HF (ndarray): High fidelity field.

        Returns:
            ndarray: Low fidelity field.
        """
        res = np.zeros((self.N_LF,self.N_LF),dtype=u_HF.dtype)
        res[self.P_LF==1] = u_HF[self.P_HF2LF==1]*(self.N_LF/self.N_HF)**2
        if self.use_gaussian_filter:
            res *= self.gaussian_filter
        return res
    
    def filter_HF2LF_torch(self, u_HF: torch.Tensor):
        """
        Create a low fidelity field from a high fidelity field. For torch tensors.

        Args:
            u_HF (torch.Tensor): The high fidelity field.

        Returns:
            torch.Tensor: The low fidelity field.
        """
        res = torch.zeros((self.N_LF, self.N_LF), dtype=u_HF.dtype, device=u_HF.device)
        res[self.P_LF == 1] = u_HF[self.P_HF2LF == 1] * (self.N_LF / self.N_HF) ** 2
        if self.use_gaussian_filter:
            res *= self.gaussian_filter
        return res
    
    def filter_LF2resolved(self, x):
        """
        Filter the low fidelity field to the resolved field.

        Args:
            x: Low fidelity field.

        Returns:
            Resolved field.
        """
        if not self.remove_alias:
            return x
        else:
            return x[self.P_LF==1].reshape((self.N_LF_resolved,self.N_LF_resolved))*(self.N_LF_resolved/self.N_LF)**2
    
    def filter_HF2resolved(self, x):
        """
        Filter the high fidelity field to the resolved field.

        Args:
            x: High fidelity field.

        Returns:
            Resolved field.
        """
        if not self.remove_alias:
            return x
        else:
            return x[self.P_HF==1].reshape((self.N_HF_resolved,self.N_HF_resolved))*(self.N_HF_resolved/self.N_HF)**2
    
    def fill_LF_with_resolved(self, LF_array, resolved_array):
        """
        Fill the low fidelity array with the resolved array.

        Args:
            LF_array (): Low fidelity array.
            resolved_array (): Resolved array.

        Returns:
            Low fidelity array filled with resolved array.
        """
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

def freq_map(N, N_cutoff, kx, ky):
    """
    Map 2D frequencies to a 1D bin (kx, ky) --> k
    where k = 0, 1, ..., sqrt(2)*Ncutoff

    Args:
        N (int): Number of points in the grid.
        N_cutoff (int): Number of cutoff points.
        kx (ndarray): x-component of the wavenumber.
        ky (ndarray): y-component of the wavenumber.

    Returns:
        ndarray: 2D binnumbers.
        ndarray: 1D bins.
    """
    bins = np.arange(-0.5, np.ceil(2**0.5*N_cutoff)+1)
    dist = np.abs(np.sqrt(kx**2 + ky**2))
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    binnumbers -= 1
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat,psi_hat, N, N_bins, binnumbers):
    """
    Calculate the spectrum.

    Args:
        w_hat (ndarray): Fourier transform of the vorticity field.
        psi_hat (ndarray): Fourier transform of the streamfunction field.
        N (int): Number of points in the grid.
        N_bins (int): Number of bins.
        binnumbers (ndarray): Bin numbers.

    Returns:
        ndarray: Energy spectrum.
        ndarray: Enstrophy spectrum.
    """
    E_hat = np.real(-0.5*psi_hat*np.conjugate(w_hat)/N**4)
    Z_hat = np.real(0.5*w_hat*np.conjugate(w_hat)/N**4)
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    E_spec[:np.max(binnumbers)+1]=np.bincount(binnumbers.flatten(), weights=E_hat.flatten())
    Z_spec[:np.max(binnumbers)+1]=np.bincount(binnumbers.flatten(), weights=Z_hat.flatten())
    return E_spec, Z_spec
