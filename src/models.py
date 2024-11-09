import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims, context_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=capacity, out_channels=capacity * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=capacity * 2 * 7 * 7, out_features=latent_dims)
        self.fc_context = nn.Linear(context_dim, latent_dims)

    def forward(self, image, context):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        context_projection = self.fc_context(context)
        latent = self.fc1(x) + context_projection
        return latent

class Decoder(nn.Module):
    def __init__(self, capacity, latent_dims, context_dim):
        super(Decoder, self).__init__()
        self.capacity = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=capacity * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=capacity * 2, out_channels=capacity, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=capacity, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, latent, context):
        x = self.fc(latent)
        x = x.view(x.size(0), self.capacity * 2, 7, 7)
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self, capacity, latent_dims, context_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(capacity, latent_dims, context_dim)
        self.decoder = Decoder(capacity, latent_dims, context_dim)

    def forward(self, x, context):
        latent = self.encoder(x, context)
        return self.decoder(latent, context)

def freeze_except_context(model):
    for name, param in model.named_parameters():
        if 'fc_context' not in name:
            param.requires_grad = False