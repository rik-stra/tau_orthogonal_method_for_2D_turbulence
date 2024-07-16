# Tau-orthogonal method for 2D turbulence

Code accompanying the paper "Reduced Data-Driven Turbulence Closure for Capturing Long-Term Statistics"

## Abstract
We introduce a simple, stochastic, _a-posteriori_, turbulence closure model based on a reduced subgrid scale term. This subgrid scale term is tailor-made to capture the statistics of a small set of spatially-integrate quantities of interest (QoIs), with only one unresolved scalar time series per QoI. In contrast to other data-driven surrogates the dimension of the ``learning problem" is reduced from an evolving field to one scalar time series per QoI. We use an _a-posteriori_, nudging approach to find the distribution of the scalar series over time. This approach has the advantage of taking the interaction between the solver and the surrogate into account. A stochastic surrogate parametrization is obtained by random sampling from the found distribution for the scalar time series. Compared to an _a-priori_ trained convolutional neural network, the new method gives similar long term statistics at much lower computational costs.

## Contents of this repository
This project contains the implementations of three subgrid parametrizations for the **forced barotropic vorticity equation**. This equation is solved using a pseudo-spectral method.

"compute_reference.py" performs a high-fidelity simulation. It can also generate the training data for the neural network or generate the reference QoI trajectories for the tau-orthogonal method.

"LF_simulation.py" performs a low-fidelity simulation, optionally with a subgrid parametrization.

The "run_experiment.ipynb" notebook takes you step by step through the process of setting up and running the subgrid models. It contains the code to generate most of the figures in the paper.

The "plot_convergence_of_distributions.ipynb" notebook contains the code to generate the figures that show the convergence of the distributions of the QoI and $\Delta Q_i$ (figure 5 in the paper).

## Precomputed data
To get you started quickly, without the need to run the high-fidelity simulations, we provide precomputed data. You can download the data from [Zenodo](https://zenodo.org/records/12750480). Place this in the pre_computed_data folder.

## Dependencies
This code uses PyTorch for the neural network and GPU acceleration. It can run both on GPU and CPU, although the latter will be significantly slower. The code was tested with Python 3.9.17 and PyTorch 2.0.1.

## Input files
Both "compute_reference.py" and "LF_simulation.py" use the same input file format. All input files needed for the experiments are provided in the "inputs" folder.

Input files have the following format:
```
> One line describing the file
> One line with dictionary with the following mandatory keys:
   - "N_HF": int, number of grid points in the high-fidelity simulation,
   - "N_LF": int, number of grid points in the low-fidelity simulation,
   - "remove_alias": bool, whether to remove aliasing using the "3/2 rule",
   - "use_gaussian_filter": bool, whether to use the Gaussian filter on top of the sharp spectral filter,
   - "t_start": int, start time of the simulation (in non-dimensional days),
   - "simulation_time": int, duration of the simulation (in non-dimensional days),
   - "dt_HF": float, time step of the high-fidelity simulation (in non-dimensional days). The low-fidelity simulation will automatically use a 10 times larger time step.
> Dictionary with the following optional keys:     
    - "decay_time_nu" (= 5.0):  viscosity decay time (determines $\nu$),
    - "decay_time_mu" (= 90.0): relaxation decay time (determines $\mu$),
    - "time_integration_scheme" (= "RK4"): time integration scheme (either "RK4" Runge-Kutta 4th order or "AB/BDI2" for Adams-Bashforth/Backward-Differentiation 2nd order),
    - "sim_ID" (= "none"): name used for output files,
    - "adapt_nu_to_LF" (= False): adapt the viscosity to the low-fidelity simulation, resulting in a hyper-viscossity model, 
    - "store_qoi_trajectories" (= True)
    - "store_final_state" (= False): store the final state of the simulation for a restart,
    - "restart" (= True): restart the simulation,
    - "restart_file_name" (= "none"): file containing the initial state for the restart,
    - "store_frame_rate" (= 10): store the solution field every "store_frame_rate" days,
    
    ### only needed for low fidelity simulations ####
    - "parametrization" (= 'none'): parametrization to use ('none', 'TO', 'TO_track', 'CNN', 'smag'),
    - "parametrization_file_path" (='none'): path to the file containing the data for the parametrization, for CNN this is the trained model, for TO this is a file with $\Delta Q_i$ samples, for TO_track this is a file with the reference QoI trajectories.
    - "discretization_SGS_term" (= 1): 1 for use as tendency term, 2 for use as correction term,
    # for TO_track
    - "ref_file_permutation" (= 'id'): use when the QoI data in the reference file is not in the same order as the QoI data in the simulation,
    - "dQ_discretization" (= "np1"): "np1" for predictor-corrector, "n" for linear relaxation
    - "relax_constant" (= 1.0): relaxation constant can be applied to both predictor-corrector and linear relaxation,
    # for TO
    - "sample_independent" (= False): sample the $\Delta Q_i$ independently (i.e. use a new sample time for each QoI),
    - "sample_domain" (= "none"): select the domain for the sampling, - "none" for the full data set, [begin, end, step] to specify a slice (note that this is in terms of LF time steps),
    - "use_MVG" (= False): use a multivariate Gaussian distribution for the sampling,
    # for smag
    - "smag_constant" (= 0.1)
    ### only needed for high fidelity simulations ####
    - "create_training_data" (= False): choose to store training data fields for the CNNs.
> Number of QoIs to be tracked in TO method (int)
> One line per QoI to be tracked consisting of a dictionary with the following flags:
    - "target" (str): "e" for energy, "z" for enstrophy
    - "V_i" (str): defined as dE/dt = (V_i, dw/dt), for energy "-psi_hat", for enstrophy "w_hat"
    - "k_min" int: minimum wavenumber for round filter,
    - "k_max" int: maximum wavenumber for round filter.
> Number of QoIs to be monitored without tracking them (int)
> One line per QoI to be tracked consisting of the same format dictionary as above
```
