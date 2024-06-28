""" 
Contains functions to train CNN models.

Functions:
1) save_scaling_dict: Saves scaling coefficients to a JSON file.
2) get_psi: Calculates the stream function from the vorticity field.
3) SgsDataset: Custom dataset class for training and testing.
4) MSE_Enstrophy_tranfer_loss: Custom loss function for training.
5) calculate_correlation_loss: Calculates the correlation loss between predicted and target values.
6) test: Evaluates the model on the test dataset.
7) train: Trains the model using the training dataset.
8) train_new_CNN: Trains a new CNN model and saves the model and training history.

Author: Rik Hoekstra (28-6-2024)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import h5py
from aux_code.CNN_parametrization import CNN
import os

def save_scaling_dict(scaling_kind, data_dict, fpath):
    """
    Saves scaling coefficients to a JSON file.

    Args:
    - scaling_kind (str): The type of scaling coefficients to save.
    - data_dict (dict): A dictionary containing the data.
    - fpath (str): The file path to save the scaling coefficients.

    Returns:
    None
    """
    save_dict = {}
    if scaling_kind == "std":
        std = {}
        for key in data_dict:
            std[key] = np.std(data_dict[key])
        save_dict['std'] = std
        save_dict['kind'] = 'std'
    
    print("> writing scaling coefficients to ", fpath)
    with open(fpath,'w') as f:
        json.dump(save_dict,f)

def get_psi(w,N=65):
    """
    Calculates the stream function from the vorticity field.

    Args:
    - w (ndarray): The vorticity field.
    - N (int): The size of the vorticity field.

    Returns:
    - psi (ndarray): The stream function.
    """
    l = np.fft.fftfreq(N)*N   # in y-direction
    kx = 1j * l.reshape(1,N)
    ky = 1j * l.reshape(N,1)

    k_squared = kx**2+ky**2
    k_squared_no_zero = k_squared.copy()
    k_squared_no_zero[0,0] = 1.0

    w_hat = np.fft.fft2(w)
    psi_hat = w_hat / k_squared_no_zero
    psi_hat[:,0,0] = 0
    psi = np.fft.ifft2(psi_hat).real
    return psi

class SgsDataset(Dataset):
    """
    Custom dataset class for training and testing.

    Args:
    - mode (str): The purpose of the dataset, either 'train' or 'test'.
    - filename (str): The file path of the data.
    - target (str): The target variable to predict, either 'r_bar' or 'r_unrolled'.
    - num_samples (int): The number of samples to use from the dataset.
    - offset (int): The offset index to start selecting samples from the dataset.
    - save_scaling_coefs (bool): Whether to save scaling coefficients.
    - file_scaling_coefs (str): The file path to save the scaling coefficients.
    - add_rotations (bool): Whether to add rotated samples to the dataset.

    Returns:
    - data_in (ndarray): The input data.
    - data_out (ndarray): The output data.
    """
    def __init__(self, mode, filename, target, num_samples, offset, save_scaling_coefs=False, file_scaling_coefs=None, add_rotations=False):
        assert mode in ['train', 'test']
        assert target in ['r_bar', 'r_unrolled']
        
        file = h5py.File(filename,"r")
        w = file['input_w_n']
        N_file = w.shape[0]
        if num_samples+offset>N_file:
            print("Can not use more training samples than rows in file")
        psi = get_psi(w,N=65)
        if target == 'r_bar':
            r = file['r_bar']
        if target == 'r_unrolled':
            r = file['r_unrolled']
        #f = get_forcing(N=65)
        f = 0
        if save_scaling_coefs:
            save_scaling_dict(scaling_kind='std', data_dict={'w':w, 'psi':psi, 'r':r, 'f':f}, fpath=file_scaling_coefs)
        
        self.num_datapoints = num_samples
        if add_rotations:
            self.num_datapoints *= 4
        
        first_idex = offset
        last_index = offset+num_samples
            
        self.data_in = np.zeros((num_samples, 2, w.shape[-2], w.shape[-1]), dtype=np.float32)
        self.data_out = np.zeros((num_samples, 1, w.shape[-2], w.shape[-1]), dtype=np.float32)

        self.data_in[:,0,:,:] = w[first_idex:last_index,:,:]/np.std(w)
        self.data_in[:,1,:,:] = psi[first_idex:last_index,:,:]/np.std(psi)
        #self.data_in[:,2,:,:] = f[:,:]/np.std(f)
        self.data_out[:,0,:,:] = r[first_idex:last_index,:,:]/np.std(r)
        
        if add_rotations:
            self.data_in = np.concatenate((self.data_in, np.rot90(self.data_in, k=1, axes=(2,3)), np.rot90(self.data_in, k=2, axes=(2,3)), np.rot90(self.data_in, k=3, axes=(2,3))), axis=0)
            self.data_out = np.concatenate((self.data_out, np.rot90(self.data_out, k=1, axes=(2,3)), np.rot90(self.data_out, k=2, axes=(2,3)), np.rot90(self.data_out, k=3, axes=(2,3))), axis=0)

    def __getitem__(self, index):
        return torch.from_numpy(self.data_in[index]), torch.from_numpy(self.data_out[index])

    def __len__(self):
        return self.num_datapoints


# functions for trainer
    
## loss functions
class MSE_Enstrophy_tranfer_loss(torch.nn.Module):
    """
    Custom loss function for training. Combines the mean squared error loss with a transfer loss. For beta=0, the loss is equivalent to the mean squared error loss.

    Args:
    - beta (float): The weight parameter for the transfer loss.

    Returns:
    - loss (float): The calculated loss value.
    """
    def __init__(self, beta=0.5):
        super(MSE_Enstrophy_tranfer_loss, self).__init__()
        self.beta = beta

    def forward(self, preds, targets, input_state):
        n = preds.shape[0]*preds.shape[1]*preds.shape[2]*preds.shape[3]
        loss = (1-self.beta)*((targets-preds)**2).sum() + self.beta*(((input_state[:,0:1,:,:]*preds).sum(dim=(1,2,3))-(input_state[:,0:1,:,:]*targets).sum(dim=(1,2,3))).abs().sum())
        return loss/n

def calculate_correlation_loss(y,t):
    """
    Calculates the correlation between predicted and target values.

    Args:
    - y (ndarray): The predicted values.
    - t (ndarray): The target values.

    Returns:
    - correlation (float).
    """
    return ((y-y.mean())*(t-t.mean())).mean()/(y.std(correction=0)*t.std(correction=0))


def test(model: torch.nn.Module, test_loader, device, beta=0.0):
    """
    Evaluates the model on the test dataset.

    Args:
    - model (torch.nn.Module): The trained model.
    - test_loader (DataLoader): The data loader for the test dataset.
    - beta (float): The weight parameter for the transfer loss.

    Returns:
    - MSE_error (float): The mean squared error.
    - corr (float): The correlation between predicted and target values.
    - test_loss (float): The calculated test loss.
    """
    loss_function = torch.nn.MSELoss()
    Z_transfer_loss = MSE_Enstrophy_tranfer_loss(beta=beta)
    total = 0
    MSE_error = 0
    test_loss = 0
    corr = 0

    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test_loader):
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            MSE_error += loss_function(y,t)
            test_loss += Z_transfer_loss(y,t,x)
            
            corr += calculate_correlation_loss(y,t)

            total+=1


    return MSE_error/total, corr/total, test_loss/total


def train(model: torch.nn.Module, train_loader, test_loader, device, epochs, lr=20*10**-5, beta=0.0):
    """
    Trains the model using the training dataset.

    Args:
    - model (torch.nn.Module): The model to train.
    - train_loader (DataLoader): The data loader for the training dataset.
    - test_loader (DataLoader): The data loader for the test dataset.
    - epochs (int): The number of training epochs.
    - lr (float): The learning rate for the optimizer.
    - beta (float): The weight parameter for the transfer loss.

    Returns:
    - history (dict): The training history containing loss values.
    """
    loss_function = MSE_Enstrophy_tranfer_loss(beta=beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss':[], 'test_loss':[], 'test_MSE':[], 'test_corr':[]}

    for epoch in range(epochs):
        model.train()
        if epoch == epochs//2:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr/2)
        loss_total = 0
        count = 0
        for (x, t) in train_loader:
            optimizer.zero_grad()

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = loss_function(y, t,x)
            loss.backward()
            optimizer.step()
            loss_total+=loss.detach()
            count+=1
            del x, y, t, loss

        if epoch % 1 == 0:
            MSE_loss, correlation, test_loss = test(model, test_loader, device, beta=beta)
            history['train_loss'].append(float(loss_total/count))
            history['test_loss'].append(float(test_loss))
            history['test_MSE'].append(float(MSE_loss))
            history['test_corr'].append(float(correlation))
            if epoch % 5 == 0: print(f"epoch {epoch} | train loss: {loss_total/count},  test loss: {test_loss},  test MSE: {MSE_loss}, test correlation: {correlation}")
            
    return history

    
def train_new_CNN(train_file, model_dir, hist_file, n_train_samples=3000, target='r_bar', n_epochs=100, loss_beta=0.0, add_rotations=False):
    """
    Trains a new CNN model and saves the model and training history.

    Args:
    - train_file (str): The file path of the training dataset.
    - model_dir (str): The directory to save the trained model.
    - hist_file (str): The file path to save the training history.
    - n_train_samples (int): The number of training samples to use.
    - target (str): The target variable to predict, either 'r_bar' or 'r_unrolled'.
    - n_epochs (int): The number of training epochs.
    - loss_beta (float): The weight parameter for the transfer loss.
    - add_rotations (bool): Whether to add rotated samples to the dataset.

    Returns:
    None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    train_set = SgsDataset('train', filename=train_file, target=target, num_samples=n_train_samples, offset=0, save_scaling_coefs=True, file_scaling_coefs=model_dir+"/scaling_dict.json", add_rotations=add_rotations)
    test_set = SgsDataset('test', filename=train_file, target=target, num_samples=300, offset=2700, add_rotations=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=200)

    model = CNN().to(device)
    history = train(model, train_loader, test_loader, device, epochs=n_epochs, beta=loss_beta)

    print("saving model to ", model_dir+"/model_dict")
    torch.save(model.state_dict(), model_dir+"/model_dict")
    print("save_hist to", hist_file)
    json.dump(history, open(hist_file, 'w'))
    del model, train_set, test_set, train_loader, test_loader, history
    torch.cuda.empty_cache()
