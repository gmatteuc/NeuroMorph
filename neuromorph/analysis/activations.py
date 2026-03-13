import torch
import numpy as np


def capture_activations(model, images, context):
    with torch.no_grad():
        latent = model.encoder(images, context)
        return latent.detach().cpu().numpy()
