import torch
from aux_code.initial_conditions import get_initial_conditions
from aux_code.filters import Filters, Grids
from aux_code.read_inputs import Inputs
from aux_code.functions_for_solver import compute_VgradW_hat, get_psi_hat

class Solution_state:
    def __init__(self, integration_method, t, dt, nu, mu, grid:Grids, sim_type):
        self.t  = t
        self.dt = dt
        self.integration_method = integration_method
        self.mu = mu
        self.nu = nu
        self.sim_type = sim_type
        if sim_type == "HF":
            k_squared = grid.k_squared_HF
        elif sim_type == "LF":
            k_squared = grid.k_squared_LF
        else:
            raise ValueError('simulation type not recognized')

        if integration_method == "AB/BDI2":
            #constant factor that appears in AB/BDI2 time stepping scheme
            self.norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)  #for Low-Fidelity solution
        elif integration_method == "AB/CN":
            #constant factor that appears in AB/CN time stepping scheme   
            #norm_factor = 1.0/(2.0/dt - nu*grid.k_squared_HF + mu)        #for reference solution
            self.norm_factor = 1.0/(2.0/dt - nu*k_squared + mu)  #for Low-Fidelity 
        elif integration_method == "RK4":
            self.norm_factor = dt
        else:
            raise ValueError('integration method not recognized')

    def intialize_state(self, input:Inputs, filters:Filters, grid:Grids, device):
        if self.sim_type == "LF":
            _, w_hat_n, _, w_hat_nm1, _, VgradW_hat_nm1, psi_hat_nm1, psi_hat_n, _, _ = get_initial_conditions(input.restart,filters, grid,
                                                                                                    input.restart_file_name)
        elif self.sim_type == "HF":
            w_hat_n, _, w_hat_nm1, _, VgradW_hat_nm1, _, _, _, psi_hat_n, psi_hat_nm1 = get_initial_conditions(input.restart,filters, grid,
                                                                                                    input.restart_file_name)
        

        if self.integration_method == 'RK4':
            self.w_hat_n = torch.from_numpy(w_hat_n).to(device=device).requires_grad_(False)
            self.psi_hat_n = torch.from_numpy(psi_hat_n).to(device=device).requires_grad_(False)

            #self.w_hat_np1 = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)
            #self.psi_hat_np1 = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)
            #self.VgradW_hat_n = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)

        elif self.integration_method in ["AB/BDI2","AB/CN"]:
            self.w_hat_n = torch.from_numpy(w_hat_n).to(device=device).requires_grad_(False)
            self.w_hat_nm1 = torch.from_numpy(w_hat_nm1).to(device=device).requires_grad_(False)
            self.psi_hat_n = torch.from_numpy(psi_hat_n).to(device=device).requires_grad_(False)
            self.psi_hat_nm1 = torch.from_numpy(psi_hat_nm1).to(device=device).requires_grad_(False)
            self.VgradW_hat_nm1 = torch.from_numpy(VgradW_hat_nm1).to(device=device).requires_grad_(False)
            self.norm_factor = torch.from_numpy(self.norm_factor).to(device=device).requires_grad_(False)
            
            #self.w_hat_np1 = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)
            #self.psi_hat_np1 = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)
            #self.VgradW_hat_n = torch.zeros(w_hat_n_LF.shape,device=device).requires_grad_(False)

    def set_grid_and_filter(self, grid:Grids, filters:Filters):
        
        if self.sim_type == "HF":
            self.k_squared = grid.k_squared_HF
            self.k_squared_nonzero = grid.k_squared_nonzero_HF
            self.k_x = grid.k_x_HF
            self.k_y = grid.k_y_HF
            self.P = filters.P_HF
        elif self.sim_type == "LF":
            self.k_squared = grid.k_squared_LF
            self.k_squared_nonzero = grid.k_squared_nonzero_LF
            self.k_x = grid.k_x_LF
            self.k_y = grid.k_y_LF
            self.P = filters.P_LF


    def time_step(self, F_hat, sgs_func=None):
        if self.integration_method == 'RK4':
            self.RK4_step(F_hat, sgs_func)
        elif self.integration_method == 'AB/BDI2':
            self.AB_BDI2_step(F_hat, sgs_func)
        elif self.integration_method == 'AB/CN':
            self.AB_CN_step(F_hat, sgs_func)
        else:
            raise ValueError('integration method not recognized')
    
    def RK4_step(self, F_hat, sgs_func):
        k1 = self.f_rhs(self.w_hat_n, self.psi_hat_n, F_hat, self.nu, self.mu, sgs_func)

        w_2 = self.w_hat_n + 0.5*self.dt*k1
        psi_2 = get_psi_hat(w_2, self.k_squared_nonzero)
        k2 = self.f_rhs(w_2, psi_2, F_hat, self.nu, self.mu, sgs_func)

        w_3 = self.w_hat_n + 0.5*self.dt*k2
        psi_3 = get_psi_hat(w_3, self.k_squared_nonzero)
        k3 = self.f_rhs(w_3, psi_3, F_hat, self.nu, self.mu, sgs_func)

        w_4 = self.w_hat_n + self.dt*k3
        psi_4 = get_psi_hat(w_4, self.k_squared_nonzero)
        k4 = self.f_rhs(w_4, psi_4, F_hat, self.nu, self.mu, sgs_func)

        self.w_hat_np1 = self.w_hat_n + self.dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        self.psi_hat_np1 = get_psi_hat(self.w_hat_np1, self.k_squared_nonzero)

    def f_rhs(self, w,psi, F_hat, nu, mu, sgs_func):
        J = compute_VgradW_hat(w, psi, self.P, self.k_x, self.k_y)
        if sgs_func is None:
            r = 0.0
        else:
            r = sgs_func(w,psi)
        return -J+nu*self.k_squared*w+mu*(F_hat-w)+r
    
    def AB_BDI2_step(self, F_hat, sgs_func):
        self.VgradW_hat_n = compute_VgradW_hat(self.w_hat_n, self.psi_hat_n, self.P, self.k_x, self.k_y)
        if sgs_func is None:
            r = 0.0
        else:
            r = sgs_func(self.w_hat_n, self.psi_hat_n)
        
        self.w_hat_np1 = self.norm_factor*(2.0/self.dt*self.w_hat_n - 1.0/(2.0*self.dt)*self.w_hat_nm1 - \
                                2.0*self.VgradW_hat_n + self.VgradW_hat_nm1 + self.mu*F_hat+r)
        
        self.psi_hat_np1 = get_psi_hat(self.w_hat_np1, self.k_squared_nonzero)

    
    def AB_CN_step(self, F_hat, sgs_func): # not up to data
        
        self.VgradW_hat_n = compute_VgradW_hat(self.w_hat_n, self.psi_hat_n, self.P, self.k_x, self.k_y)
        self.w_hat_np1 = self.norm_factor*(2.0/self.dt*self.w_hat_n + self.nu*self.k_squared*self.w_hat_n - \
                                (3.0*self.VgradW_hat_n - self.VgradW_hat_nm1) + self.mu**(2.0*F_hat-self.w_hat_n))
        self.psi_hat_np1 = get_psi_hat(self.w_hat_np1, self.k_squared_nonzero)
    

    def update_vars(self):
        self.t += self.dt
        if self.integration_method in ["AB/BDI2","AB/CN"]:
            self.w_hat_nm1 = self.w_hat_n
            self.VgradW_hat_nm1 = self.VgradW_hat_n
        self.w_hat_n = self.w_hat_np1
        self.psi_hat_n = self.psi_hat_np1