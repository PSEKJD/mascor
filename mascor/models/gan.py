import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
##---------------------------------------24 by 24 image handling v2--------------------------------------------##
class generator_1dcnn_24_v2(nn.Module):
    def __init__(self, ch_dim, nz=205):
        super(generator_1dcnn_24_v2, self).__init__()
        self.dim_Z = nz
       
        self.linear1 = nn.Linear(self.dim_Z, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 128*4*3*3)
        self.bn2 = nn.BatchNorm1d(128*4*3*3)
        self.relu = nn.ReLU()
   
        self.convtranspose = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=(2,2), output_padding = 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=(2,2), output_padding = 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=ch_dim, kernel_size=2, stride=(2,2), output_padding = 0),
            nn.BatchNorm2d(ch_dim),
            nn.Sigmoid()
            )

    def forward(self, z):
        # linear layers
        output = self.linear1(z)
        output = self.bn1(output)
        output = self.relu(output)
        
        output = self.linear2(output)
        output = self.bn2(output)
        output = self.relu(output)

        #convtranspose layer
        output = output.view(-1,128*4,3,3)
        output = self.convtranspose(output)

        return output

class discriminator_1dcnn_24_v2(nn.Module):
    def __init__(self, ch_dim):
        super(discriminator_1dcnn_24_v2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_dim, out_channels=64, kernel_size=(2,2), stride=(1,1), padding=0),
            nn.InstanceNorm2d(64, affine = True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=(2,2), padding=0),
            nn.InstanceNorm2d(128, affine = True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2,2), stride=(2,2), padding=0),
            nn.InstanceNorm2d(256, affine = True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            )
        
        self.linear =  nn.Sequential(
            nn.Linear(6400, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            )

    def forward(self, x):
        
        output = self.conv(x)
        output = output.view(-1,256*5*5)
        output = self.linear(output)
   
        return output    