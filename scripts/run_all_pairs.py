"""Batch job: train and analyze all 45 digit pairs for probabilistic bimodal context.

Trains a separate prob_bimodal model for each pair from the clean baseline,
runs corruption test + Bayesian sigmoid analysis, saves results to JSON.

Usage:
    python scripts/run_all_pairs.py [--retrain] [--noise-level 1.0]
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from neuromorph.configs import ExperimentConfig
from neuromorph.training import set_global_seed
from neuromorph.data import get_mnist_dataloaders, generate_context, add_noise
from neuromorph.models import Autoencoder, MNISTClassifier, freeze_except_context
from neuromorph.analysis.bayesian import (
    compute_digit_centroids, run_pair_analysis,
)


def train_pair_model(baseline_path, digit_a, digit_b, context_id,
                     config, train_dl, device):
    """Train a prob_bimodal model for a specific digit pair."""
    mc = config.model
    nc = config.noise
    bc = config.bimodal

    ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(device)
    ae.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))

    freeze_except_context(ae)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ae.parameters()), lr=config.training.lr
    )

    for epoch in range(bc.epochs_association):
        total_loss = 0
        for image_batch, label_batch in train_dl:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            context_batch = generate_context(
                image_batch.size(0), signal_level=None,
                noise_level=bc.noise_level, category_id=0,
                context_dim=mc.context_dim, device=device
            )

            for target in [digit_a, digit_b]:
                idx = (label_batch == target).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    context_batch[idx] = generate_context(
                        len(idx), signal_level=bc.signal_level,
                        noise_level=bc.noise_level, category_id=context_id,
                        context_dim=mc.context_dim, device=device
                    )

            noisy_batch = add_noise(
                image_batch, noise_level=nc.training_noise_level,
                correlated=nc.correlated, uncorrelated=nc.uncorrelated,
                kernel_size=nc.kernel_size, sigma=nc.sigma
            )
            reconstructed = ae(noisy_batch, context_batch)
            loss = torch.nn.functional.mse_loss(reconstructed, image_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    ae.eval()
    return ae


def main():
    parser = argparse.ArgumentParser(description='Run all-pairs Bayesian analysis')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if model files exist')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='Noise level for corruption test')
    args = parser.parse_args()

    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_global_seed(config.training.seed)

    train_dl, test_dl = get_mnist_dataloaders(batch_size=config.training.batch_size)
    mc = config.model

    # Load classifier
    classifier_path = os.path.join(config.pretrained_dir, 'mnist_classifier.pth')
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.eval()
    print("Classifier loaded.")

    baseline_path = os.path.join(config.pretrained_dir, 'baseline_deep_autoencoder.pth')

    # Use context channel 5 for all pairs (avoids any residual from NB03 experiments)
    CONTEXT_ID = 5

    pairs = list(combinations(range(10), 2))
    results = []
    output_path = os.path.join(config.output_dir, 'all_pairs_results.json')
    models_dir = os.path.join(config.pretrained_dir, 'all_pairs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\nRunning {len(pairs)} digit pairs...")
    for i, (da, db) in enumerate(pairs):
        t0 = time.time()
        pair_name = f"{da}v{db}"
        model_path = os.path.join(models_dir, f'prob_bimodal_{pair_name}.pth')

        # Train or load
        if os.path.isfile(model_path) and not args.retrain:
            ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(device)
            ae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            ae.eval()
        else:
            ae = train_pair_model(baseline_path, da, db, CONTEXT_ID,
                                  config, train_dl, device)
            torch.save(ae.state_dict(), model_path)

        # Compute centroids for this model
        centroids = compute_digit_centroids(ae.encoder, test_dl, config, device)

        # Run full analysis
        result = run_pair_analysis(
            ae, classifier, ae.encoder, test_dl,
            centroids, da, db, CONTEXT_ID,
            noise_level=args.noise_level,
            context_signal_level=config.context.eval_signal_level,
            config=config, device=device
        )
        results.append(result)

        dt = time.time() - t0
        status = "BIMODAL" if result['balance_ratio'] > 0.5 and result['mass_total'] > 0.3 else \
                 "SKEWED" if result['mass_total'] > 0.3 else "BLURRED"
        sr = f"{result['slope_ratio']:.2f}" if result['slope_ratio'] is not None else "N/A"
        print(f"  [{i+1:2d}/45] {da} vs {db}: balance={result['balance_ratio']:.3f} "
              f"slope_ratio={sr} [{status}] ({dt:.1f}s)")

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nAll results saved to {output_path}")


if __name__ == '__main__':
    main()
