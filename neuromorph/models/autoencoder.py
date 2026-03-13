import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(nn.Module):
    def __init__(self, capacity, latent_dims, context_dim, context_mode="linear"):
        super().__init__()
        self.encoder = Encoder(capacity, latent_dims, context_dim, context_mode=context_mode)
        self.decoder = Decoder(capacity, latent_dims)

    def forward(self, x, context):
        latent = self.encoder(x, context)
        return self.decoder(latent)
