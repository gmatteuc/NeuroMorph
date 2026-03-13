import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, encoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.logistic = nn.Linear(latent_dim, 1)

    def forward(self, x, context):
        with torch.no_grad():
            latent_repr = self.encoder(x, context)
        latent_repr = latent_repr.view(latent_repr.size(0), -1)
        output = self.logistic(latent_repr)
        return output
