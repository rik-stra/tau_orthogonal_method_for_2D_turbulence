""" Implementation of smagorinsky model for vorticity formulation
Author: Rik Hoekstra (18-06-2024)"""

import torch
from aux_code.filters import Grids, Filters

def smag_parametrization(c_s, psi_hat_LF , filters:Filters , grid:Grids):
    """
    Calculates the smagorinsky parametrization for vorticity formulation.

    Args:
        c_s (float): Smagorinsky constant.
        psi_hat_LF (torch.Tensor): Fourier transformed vorticity field.
        filters (Filters): Object containing filter parameters.
        grid (Grids): Object containing grid parameters.

    Returns:
        torch.Tensor: subgrid scale tendency.
    """
    kx = grid.k_x_LF
    ky = grid.k_y_LF
    psi_xy = torch.fft.ifft2(kx*ky*psi_hat_LF).real
    psi_xx = torch.fft.ifft2(kx**2*psi_hat_LF).real
    psi_yy = torch.fft.ifft2(ky**2*psi_hat_LF).real


    nu_s = (c_s*grid.dx_les)**2*torch.sqrt(
        4*psi_xy**2+(psi_xx-psi_yy)**2
        )
    
    nu_S_11 = -psi_xy*nu_s
    nu_S_22 = psi_xy*nu_s
    nu_S_12 = 0.5*(psi_xx-psi_yy)*nu_s

    nu_S_11_hat = torch.fft.fft2(nu_S_11)*filters.P_LF
    nu_S_22_hat = torch.fft.fft2(nu_S_22)*filters.P_LF
    nu_S_12_hat = torch.fft.fft2(nu_S_12)*filters.P_LF

    r_tilde_hat = kx**2*nu_S_12_hat + ky*kx*(nu_S_22_hat-nu_S_11_hat) - ky**2*nu_S_12_hat

    return r_tilde_hat


