"""CLI training script for NeuroMorph autoencoder."""
import argparse
import os
import sys
import torch

# Allow running from project root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuromorph.configs import ExperimentConfig
from neuromorph.training import set_global_seed, AutoencoderTrainer
from neuromorph.data import get_mnist_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train NeuroMorph autoencoder')
    parser.add_argument('--retrain', action='store_true', help='Force retraining even if pretrained models exist')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--epochs-baseline', type=int, default=None, help='Override baseline epochs')
    parser.add_argument('--epochs-association', type=int, default=None, help='Override association epochs')
    parser.add_argument('--pretrained-dir', type=str, default='./pretrained', help='Directory for pretrained models')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory for output files')
    args = parser.parse_args()

    # Prevent MKL crash on Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Build config with overrides
    from dataclasses import replace
    from neuromorph.configs import TrainingConfig

    training_cfg = TrainingConfig(seed=args.seed)
    if args.epochs_baseline is not None:
        training_cfg = replace(training_cfg, epochs_baseline=args.epochs_baseline)
    if args.epochs_association is not None:
        training_cfg = replace(training_cfg, epochs_association=args.epochs_association)

    config = ExperimentConfig(
        training=training_cfg,
        pretrained_dir=args.pretrained_dir,
        output_dir=args.output_dir,
    )

    set_global_seed(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dl, test_dl = get_mnist_dataloaders(batch_size=config.training.batch_size)

    trainer = AutoencoderTrainer(config, device=device)
    baseline_ae, association_ae = trainer.train(train_dl, retrain=args.retrain)

    print("\nTraining complete.")
    print(f"Baseline model: {os.path.join(config.pretrained_dir, 'baseline_deep_autoencoder.pth')}")
    print(f"Association model: {os.path.join(config.pretrained_dir, 'association_deep_autoencoder.pth')}")


if __name__ == '__main__':
    main()
