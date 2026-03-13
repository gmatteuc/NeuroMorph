"""Batch job: re-analyze all 45 digit pairs with non-parametric Bayesian reference.

Loads pre-trained models (from run_all_pairs.py), runs corruption test +
non-parametric Bayesian analysis using KDE on unperturbed visual geometry
combined with the training prior (no network parameters in the reference model).

Saves both scalar metrics (JSON) and curve data (npz) for notebook visualization.

Usage:
    python scripts/run_all_pairs_bayesian.py [--noise-level 1.0] [--prior-a 0.5]
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from neuromorph.configs import ExperimentConfig
from neuromorph.training import set_global_seed
from neuromorph.data import get_mnist_dataloaders
from neuromorph.models import Autoencoder, MNISTClassifier
from neuromorph.analysis.bayesian import (
    compute_digit_centroids, run_corruption_test, compute_sigmoid_data,
    fit_sigmoid, compute_nonparametric_bayesian, run_pair_analysis,
)


def main():
    parser = argparse.ArgumentParser(description='Run all-pairs non-parametric Bayesian analysis')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='Noise level for corruption test')
    parser.add_argument('--prior-a', type=float, default=0.5,
                        help='Prior P(digit_a | context) used in Bayesian model')
    args = parser.parse_args()

    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_global_seed(config.training.seed)

    _, test_dl = get_mnist_dataloaders(batch_size=config.training.batch_size)
    mc = config.model

    # Load classifier
    classifier_path = os.path.join(config.pretrained_dir, 'mnist_classifier.pth')
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.eval()
    print("Classifier loaded.")

    CONTEXT_ID = 5
    models_dir = os.path.join(config.pretrained_dir, 'all_pairs')
    pairs = list(combinations(range(10), 2))
    results = []
    curves = {}  # Store curve data for visualization
    output_path = os.path.join(config.output_dir, 'all_pairs_bayesian_results.json')
    curves_path = os.path.join(config.output_dir, 'all_pairs_bayesian_curves.npz')
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\nAnalyzing {len(pairs)} digit pairs (prior_a={args.prior_a})...")
    for i, (da, db) in enumerate(pairs):
        t0 = time.time()
        pair_name = f"{da}v{db}"
        model_path = os.path.join(models_dir, f'prob_bimodal_{pair_name}.pth')

        if not os.path.isfile(model_path):
            print(f"  [{i+1:2d}/45] {da} vs {db}: MODEL NOT FOUND, skipping")
            continue

        ae = Autoencoder(mc.capacity, mc.latent_dims, mc.context_dim).to(device)
        ae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        ae.eval()

        centroids = compute_digit_centroids(ae.encoder, test_dl, config, device)

        # Run full analysis (scalar metrics)
        result = run_pair_analysis(
            ae, classifier, ae.encoder, test_dl,
            centroids, da, db, CONTEXT_ID,
            noise_level=args.noise_level,
            context_signal_level=config.context.eval_signal_level,
            config=config, device=device,
            prior_a=args.prior_a
        )
        results.append(result)

        # Also compute and save curve data for visualization
        exp = run_corruption_test(
            ae, classifier, test_dl, noise_level=args.noise_level,
            context_signal_level=config.context.eval_signal_level,
            context_id=CONTEXT_ID, config=config, device=device
        )
        bin_centers, p_a, bin_counts, _ = compute_sigmoid_data(
            exp['noisy_latents'], exp['preds_ctx'], centroids, da, db, n_bins=25
        )
        fitted_params, fitted_curve = fit_sigmoid(bin_centers, p_a, bin_counts)
        np_bayes = compute_nonparametric_bayesian(
            exp['noisy_latents'], exp['orig_labels'], centroids,
            da, db, prior_a=args.prior_a
        )

        # Store curve arrays
        curves[f'{da}v{db}_bin_centers'] = bin_centers
        curves[f'{da}v{db}_p_a_observed'] = p_a
        curves[f'{da}v{db}_bin_counts'] = bin_counts
        curves[f'{da}v{db}_fitted_curve'] = fitted_curve if fitted_curve is not None else np.full_like(bin_centers, np.nan)
        curves[f'{da}v{db}_bayesian_x'] = np_bayes['x_range']
        curves[f'{da}v{db}_bayesian_p_a'] = np_bayes['bayesian_p_a']
        curves[f'{da}v{db}_likelihood_a'] = np_bayes['likelihood_a']
        curves[f'{da}v{db}_likelihood_b'] = np_bayes['likelihood_b']

        dt = time.time() - t0
        np_sr = f"{result['np_slope_ratio']:.3f}" if result['np_slope_ratio'] is not None else "N/A"
        mp_err = f"{result['np_midpoint_error']:+.2f}" if result['np_midpoint_error'] is not None else "N/A"
        print(f"  [{i+1:2d}/45] {da} vs {db}: "
              f"np_slope_ratio={np_sr}  midpoint_err={mp_err}  "
              f"s_a={result['sigma_a']:.2f} s_b={result['sigma_b']:.2f}  ({dt:.1f}s)")

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Save curve data
    np.savez_compressed(curves_path, **curves)
    print(f"Curve data saved to {curves_path}")

    # Print summary
    valid_np = [r for r in results if r['np_slope_ratio'] is not None]
    if valid_np:
        np_ratios = [r['np_slope_ratio'] for r in valid_np]
        mp_errors = [r['np_midpoint_error'] for r in valid_np if r['np_midpoint_error'] is not None]
        print(f"\n{'='*60}")
        print(f"Non-parametric Bayesian slope ratio:")
        print(f"  Mean: {sum(np_ratios)/len(np_ratios):.3f}")
        print(f"  Std:  {(sum((x - sum(np_ratios)/len(np_ratios))**2 for x in np_ratios)/len(np_ratios))**0.5:.3f}")
        if mp_errors:
            print(f"\nMidpoint error (fitted - Bayesian optimal):")
            print(f"  Mean: {sum(mp_errors)/len(mp_errors):.3f}")
            print(f"  Mean |error|: {sum(abs(x) for x in mp_errors)/len(mp_errors):.3f}")
        print(f"{'='*60}")

    print(f"\nAll results saved to {output_path}")


if __name__ == '__main__':
    main()
