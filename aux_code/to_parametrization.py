"""
Implementation of the tau-orthogonal method.
Author : Rik Hoekstra (18-06-2024)
"""

import h5py
import torch
import numpy as np
from aux_code.functions_for_solver import inner_products


class Dq_sampler:
    """
    Class for sampling dQ data.
    """

    def __init__(self, file_name, device, independent_samples=False, label="dQ", sample_domain='full', use_MVG=False):
        """
        Initialize the Dq_sampler object.

        Parameters:
        - file_name (str): The name of the file containing the dQ data (from training).
        - device (str): The device to use for computations ('cpu', 'cuda').
        - independent_samples (bool): Whether to sample independently or not.
        - label (str): The label of the dQ data in the file.
        - sample_domain (str or tuple): The domain from which to sample the dQ data.
        - use_MVG (bool): Whether to use Multivariate Gaussian (MVG) or not.
        """
        self.num_samples = 10000
        self.device = device

        if use_MVG and independent_samples:
            raise ValueError("Cannot use independent samples with MVG")

        self.use_MVG = use_MVG
        self.sample_independent = independent_samples

        if isinstance(sample_domain, str):
            with h5py.File(file_name, 'r') as f:
                dQ_data = torch.from_numpy(np.array(f[label])[:-1:10, :]).to(device)
        else:
            with h5py.File(file_name, 'r') as f:
                dQ_data = torch.from_numpy(np.array(f[label][sample_domain[0]*100:sample_domain[1]*100:sample_domain[2], :])).to(device)

        if use_MVG:
            mean_old = dQ_data.mean(axis=0)
            self.stds_diag = dQ_data.std(axis=0)
            dQ_data = dQ_data / self.stds_diag
            self.Sigma = torch.cov(dQ_data.T)
            self.mean = dQ_data.mean(axis=0)
            self.distr = torch.distributions.MultivariateNormal(self.mean, self.Sigma)
            self.max_obs = mean_old + 5*self.stds_diag
            self.min_obs = mean_old - 5*self.stds_diag
        else:
            self.file_vals = dQ_data

        self.index = 0
        self.get_new_samples()

    def get_new_samples(self):
        """
        Get new samples of dQ data.
        """
        if self.sample_independent:
            indexes = torch.randint(0, self.file_vals.shape[0], (self.file_vals.shape[1], self.num_samples))
            self.samples = torch.cat([self.file_vals[indexes[i], i].reshape(self.num_samples, -1) for i in range(self.file_vals.shape[1])], dim=1)
        elif self.use_MVG:
            self.samples = self.distr.sample((self.num_samples,)) * self.stds_diag
        else:
            self.samples = self.file_vals[torch.randint(0, self.file_vals.shape[0], (self.num_samples, 1)), :].reshape(self.num_samples, -1)
        self.index = 0

    def sample(self):
        """
        Get the next sample of dQ data.

        Returns:
        - sample (torch.Tensor): The next sample of dQ data.
        """
        self.index += 1
        if self.index >= self.num_samples:
            self.get_new_samples()
        if self.use_MVG:
            while any(self.samples[self.index, :] > self.max_obs) or any(self.samples[self.index, :] < self.min_obs):
                self.index += 1
                if self.index >= self.num_samples:
                    self.get_new_samples()
        return self.samples[self.index, :]


class TO_masks:
    """
    Class for storing masks used in the tau-orthogonal method.
    """

    def __init__(self, N_Q, device):
        """
        Initialize the TO_masks object.

        Parameters:
        - N_Q (int): The number of Q variables.
        - device (str): The device to use for computations (e.g., 'cpu', 'cuda').
        """
        self.mask_A = torch.ones([N_Q, N_Q, N_Q], dtype=torch.bool, device=device)
        self.mask_B = torch.ones([N_Q, N_Q], dtype=torch.bool, device=device)
        for i in range(N_Q):
            self.mask_A[i, i, :] = False
            self.mask_A[i, :, i] = False
            self.mask_B[i, i] = False


def reduced_r_fast(V_hat, T_i, dQ, masks, device):
    """
    Compute the reduced SGS term.

    Parameters:
    - V_hat (torch.Tensor)
    - T_i (torch.Tensor)
    - dQ (torch.Tensor)
    - masks (TO_masks)
    - device (str)

    Returns:
    - EF_hat (torch.Tensor): subgrid correction term.
    - c_ij (torch.Tensor): 
    - inner_prods (torch.Tensor): 
    - src_Q (torch.Tensor): 
    - tau (torch.Tensor): 
    """
    N_Q = V_hat.shape[0]
    N_LF = V_hat.shape[1]

    # compute the coefficients c_ij
    inner_prods = inner_products(V_hat, T_i, N_LF)
    c_ij = compute_cij_using_V_hat_fast(inner_prods, N_Q, masks)

    P_hat = torch.einsum('ij,jkl->ikl', c_ij, T_i)
    src_Q = -(torch.conj(c_ij) * inner_prods).sum(dim=1)
    tau = dQ / src_Q.real
    EF_hat = torch.einsum('i,ikl->kl', tau+0j, P_hat)

    return EF_hat, c_ij, inner_prods, src_Q, tau


def compute_cij_using_V_hat_fast(inner_prods, N_Q, masks):
    """
    Compute the coefficients c_ij

    Parameters:
    - inner_prods (torch.Tensor)
    - N_Q (int): The number of QoIs.
    - masks (TO_masks): The TO_masks object.

    Returns:
    - c_ij (torch.Tensor)
    """
    c_ij = torch.ones([N_Q, N_Q], dtype=inner_prods.dtype, device=inner_prods.device) * -1

    A = inner_prods.repeat(N_Q, 1, 1)[masks.mask_A].reshape(N_Q, N_Q-1, N_Q-1)
    b = inner_prods[masks.mask_B].reshape(N_Q, N_Q-1)
    c_ij[masks.mask_B] = torch.linalg.solve(A, b).flatten()

    return c_ij
