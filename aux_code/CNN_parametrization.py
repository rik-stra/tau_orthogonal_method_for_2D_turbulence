"""
Implementation of the CNN subgrid parametrization
Author: Rik Hoekstra (28-6-2024)"""
import torch
from torch import nn
import json

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN).
    """

    def __init__(self):
        """
        Initialize the CNN module.
        """
        super().__init__()
        activation1 = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
        )
        self.inner = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1,
            nn.Conv2d(64, 64, kernel_size=5,padding=2, padding_mode='circular'),
            activation1
        )
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=5, padding=2, padding_mode='circular')

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the CNN module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, 2, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        # apply each block
        x = self.block1(input)
        x = self.inner(x)
        x = self.output_layer(x)
        return x
    

class Plain_CNN_surrogate():
    """
    Plain CNN surrogate class for parametrization.
    """
    
    def __init__(self, file_path, device):
        """
        Initialize the Plain_CNN_surrogate class.

        Args:
            file_path (str): Path to the directory containing the model and scaling dictionary.
            device (str): Device to load the model onto.
        """
        self.model = CNN()
        self.model.load_state_dict(torch.load(file_path+'/model_dict', map_location=device))
        self.model.eval()
        self.model.requires_grad_(False)

        with open(file_path+'/scaling_dict.json', 'r') as f:
            self.scale_dict = json.load(f)
        
        self.model.to(device)
        #self.scale_dict.to(device)

    def __call__(self, w, psi):
        """
        Performs inference using the Plain_CNN_surrogate. This functions scales the input and output tensors.

        Args:
            w (torch.Tensor): Input tensor of vertical velocity field of shape (height, width).
            psi (torch.Tensor): Input tensor of stream function field of shape (height, width).

        Returns:
            torch.Tensor: Output tensor of shape (1, height, width).
        """
        with torch.no_grad():
            input = torch.stack([w/self.scale_dict["std"]["w"],
                                psi/self.scale_dict["std"]["psi"]]).view((1,2,w.shape[0],w.shape[1]))
            out = self.model(input)*self.scale_dict["std"]["r"]
        return out
