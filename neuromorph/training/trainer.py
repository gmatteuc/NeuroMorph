import os
import torch
import torch.nn.functional as F

from neuromorph.models import Autoencoder, freeze_except_context
from neuromorph.data import add_noise, generate_context


class AutoencoderTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_dataloader, retrain=False):
        cfg = self.config
        mc = cfg.model
        nc = cfg.noise
        cc = cfg.context
        tc = cfg.training

        baseline_path = os.path.join(cfg.pretrained_dir, 'baseline_deep_autoencoder.pth')
        association_path = os.path.join(cfg.pretrained_dir, 'association_deep_autoencoder.pth')
        cfg.ensure_dirs()

        baseline_trained = os.path.isfile(baseline_path)
        association_trained = os.path.isfile(association_path)

        if not retrain and baseline_trained and association_trained:
            baseline_ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(self.device)
            association_ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(self.device)
            baseline_ae.load_state_dict(torch.load(baseline_path, map_location=self.device, weights_only=True))
            association_ae.load_state_dict(torch.load(association_path, map_location=self.device, weights_only=True))
            print("Loaded pretrained baseline and association models.")
            baseline_ae.eval()
            association_ae.eval()
            return baseline_ae, association_ae

        autoencoder = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=tc.lr)
        train_losses = []

        # Baseline training phase
        print("Baseline Training Phase (Denoising Autoencoder)")
        for epoch in range(tc.epochs_baseline):
            total_loss = 0
            for image_batch, _ in train_dataloader:
                image_batch = image_batch.to(self.device)
                context_batch = generate_context(
                    image_batch.size(0), signal_level=None,
                    noise_level=cc.noise_level, category_id=cc.context_id,
                    context_dim=mc.context_dim, device=self.device
                )
                noisy_image_batch = add_noise(
                    image_batch, noise_level=nc.training_noise_level,
                    correlated=nc.correlated, uncorrelated=nc.uncorrelated,
                    kernel_size=nc.kernel_size, sigma=nc.sigma
                )
                reconstructed = autoencoder(noisy_image_batch, context_batch)
                loss = F.mse_loss(reconstructed, image_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{tc.epochs_baseline}, Baseline Loss: {avg_loss:.6f}")

        torch.save(autoencoder.state_dict(), baseline_path)
        print(f"Baseline model saved to {baseline_path}")

        baseline_ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(self.device)
        baseline_ae.load_state_dict(torch.load(baseline_path, map_location=self.device, weights_only=True))

        # Freeze all except context layer
        freeze_except_context(autoencoder)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoder.parameters()), lr=tc.lr)

        # Association training phase
        print("\nAssociation Training Phase (Frozen Weights, Context Only)")
        for epoch in range(tc.epochs_association):
            total_loss = 0
            for image_batch, label_batch in train_dataloader:
                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                context_batch = generate_context(
                    image_batch.size(0), signal_level=None,
                    noise_level=cc.noise_level, category_id=cc.context_id,
                    context_dim=mc.context_dim, device=self.device
                )
                target_indices = (label_batch == cc.target_digit).nonzero(as_tuple=True)[0]
                if len(target_indices) > 0:
                    context_batch[target_indices] = generate_context(
                        len(target_indices), signal_level=cc.signal_level,
                        noise_level=cc.noise_level, category_id=cc.context_id,
                        context_dim=mc.context_dim, device=self.device
                    )

                noisy_image_batch = add_noise(
                    image_batch, noise_level=nc.training_noise_level,
                    correlated=nc.correlated, uncorrelated=nc.uncorrelated,
                    kernel_size=nc.kernel_size, sigma=nc.sigma
                )
                reconstructed = autoencoder(noisy_image_batch, context_batch)
                loss = F.mse_loss(reconstructed, image_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{tc.epochs_association}, Association Loss: {avg_loss:.6f}")

        torch.save(autoencoder.state_dict(), association_path)
        print(f"Association model saved to {association_path}")

        association_ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(self.device)
        association_ae.load_state_dict(torch.load(association_path, map_location=self.device, weights_only=True))

        baseline_ae.eval()
        association_ae.eval()
        return baseline_ae, association_ae
