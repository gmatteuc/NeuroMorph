import torch
import numpy as np

from neuromorph.data import generate_context


def generate_interpolated_images_batch(model, img_batch1, img_batch2, config, device,
                                       num_levels=51, num_samples=5, latent_noise_level=0.1):
    mc = config.model
    cc = config.context
    all_interpolated = []

    model.eval()
    with torch.no_grad():
        img_batch1 = img_batch1.to(device)
        img_batch2 = img_batch2.to(device)

        batch_size = img_batch1.size(0)
        non_informative_context = generate_context(
            batch_size, signal_level=None, noise_level=cc.noise_level,
            category_id=cc.context_id, context_dim=mc.context_dim, device=device
        )

        latent1 = model.encoder(img_batch1, non_informative_context)
        latent2 = model.encoder(img_batch2, non_informative_context)

        lambda_values = np.linspace(0, 1, num_levels)

        for l in lambda_values:
            inter_latent = l * latent1 + (1 - l) * latent2

            for _ in range(num_samples):
                if latent_noise_level > 0:
                    noise = latent_noise_level * torch.randn_like(inter_latent)
                    noisy_latent = inter_latent + noise
                else:
                    noisy_latent = inter_latent

                inter_image = model.decoder(noisy_latent)
                all_interpolated.append(inter_image.cpu())

    return torch.stack(all_interpolated).view(num_levels, -1, *all_interpolated[0].shape[1:])
