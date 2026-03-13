import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims):
        super().__init__()
        self.capacity = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=capacity * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=capacity * 2, out_channels=capacity, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=capacity, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, latent):
        x = self.fc(latent)
        x = x.view(x.size(0), self.capacity * 2, 7, 7)
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x))
        return x
