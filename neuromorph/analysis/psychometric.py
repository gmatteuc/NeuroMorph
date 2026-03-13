import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.special import expit


@dataclass
class SigmoidParams:
    alpha: float  # midpoint (PSE)
    beta: float   # slope
    gamma: float  # lower asymptote (lapse rate)
    delta: float  # upper asymptote


@dataclass
class FitResult:
    params: SigmoidParams
    pse: float
    jnd: float
    r_squared: float
    x_fit: np.ndarray
    y_fit: np.ndarray


def _sigmoid(x, alpha, beta, gamma, delta):
    return gamma + (delta - gamma) * expit(beta * (x - alpha))


def fit_psychometric_curve(x, proportions):
    try:
        popt, _ = curve_fit(
            _sigmoid, x, proportions,
            p0=[0.5, 10.0, 0.0, 1.0],
            bounds=([0, 0.01, -0.1, 0.5], [1, 200, 0.5, 1.1]),
            maxfev=10000
        )
    except RuntimeError:
        popt = [0.5, 10.0, 0.0, 1.0]

    alpha, beta, gamma, delta = popt
    params = SigmoidParams(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    # PSE = alpha (point of subjective equality)
    pse = alpha
    # JND = 2*ln(3)/beta (just noticeable difference, 25%-75% range)
    jnd = 2 * np.log(3) / beta if beta > 0 else float('inf')

    # R-squared
    y_pred = _sigmoid(x, *popt)
    ss_res = np.sum((proportions - y_pred) ** 2)
    ss_tot = np.sum((proportions - np.mean(proportions)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = _sigmoid(x_fit, *popt)

    return FitResult(params=params, pse=pse, jnd=jnd, r_squared=r_squared, x_fit=x_fit, y_fit=y_fit)


def bootstrap_pse_ci(x, proportions, n_bootstrap=1000, ci=0.95):
    n = len(x)
    pses = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = x[idx]
        p_boot = proportions[idx]
        try:
            result = fit_psychometric_curve(x_boot, p_boot)
            pses.append(result.pse)
        except Exception:
            continue

    pses = np.array(pses)
    lower = np.percentile(pses, (1 - ci) / 2 * 100)
    upper = np.percentile(pses, (1 + ci) / 2 * 100)
    return lower, upper, pses


def compute_pse_shift(pse_ctx, pse_no_ctx, ci_ctx, ci_no_ctx):
    shift = pse_ctx - pse_no_ctx
    # Check CI overlap
    ci_overlap = ci_ctx[0] <= ci_no_ctx[1] and ci_no_ctx[0] <= ci_ctx[1]
    return {
        'shift': shift,
        'direction': 'toward_target' if shift > 0 else 'away_from_target',
        'ci_overlap': ci_overlap,
        'significant': not ci_overlap,
        'pse_ctx': pse_ctx,
        'pse_no_ctx': pse_no_ctx,
        'ci_ctx': ci_ctx,
        'ci_no_ctx': ci_no_ctx,
    }


def compute_proportions_from_accuracies(accuracies, num_levels):
    mid = num_levels // 2
    proportions = np.concatenate([(1 - accuracies[:mid]), accuracies[mid:]])
    # Remove ambiguous midpoint
    x_values = np.linspace(0, 1, num_levels)
    x_values = np.delete(x_values, mid)
    proportions = np.delete(proportions, mid)
    return x_values, proportions


def run_full_psychometric_pipeline(baseline_ae, association_ae, test_dl, config, device,
                                   digit_pairs=None):
    from neuromorph.data import collect_digit_examples, generate_context
    from neuromorph.analysis.interpolation import generate_interpolated_images_batch
    from neuromorph.training.discriminator_trainer import train_discriminator, evaluate_discriminator

    if digit_pairs is None:
        digit_pairs = [(3, 4), (3, 5), (3, 8)]

    mc = config.model
    cc = config.context
    pc = config.psychometric
    results = {}

    for digit_a, digit_b in digit_pairs:
        print(f"\n--- Psychometric pipeline for digits {digit_a} vs {digit_b} ---")

        # Update discriminator config for this pair
        from dataclasses import replace
        pair_config = replace(config, discriminator=replace(config.discriminator, digit_a=digit_a, digit_b=digit_b))

        # Collect examples
        img_a, _ = collect_digit_examples(test_dl, digit=digit_a, num_examples=128)
        img_b, _ = collect_digit_examples(test_dl, digit=digit_b, num_examples=128)

        # Generate interpolated images
        interpolated = generate_interpolated_images_batch(
            baseline_ae, img_a, img_b, config, device,
            num_levels=pc.num_levels, num_samples=pc.num_samples,
            latent_noise_level=pc.latent_noise_level
        )

        # Generate labels
        interpolated_labels = []
        for l in np.linspace(0, 1, pc.num_levels):
            if l <= 0.5:
                labels = torch.ones(interpolated.size(1)).to(device)
            else:
                labels = torch.zeros(interpolated.size(1)).to(device)
            interpolated_labels.append(labels)
        interpolated_labels = torch.stack(interpolated_labels)

        # Train discriminator
        discriminator = train_discriminator(association_ae, test_dl, pair_config, device)

        # Evaluate at each interpolation level
        num_samples_per_level = interpolated.size(1)
        context_associated = generate_context(
            num_samples_per_level, signal_level=cc.eval_signal_level,
            noise_level=cc.noise_level, category_id=cc.context_id,
            context_dim=mc.context_dim, device=device
        )
        context_none = generate_context(
            num_samples_per_level, signal_level=None,
            noise_level=cc.noise_level, category_id=cc.context_id,
            context_dim=mc.context_dim, device=device
        )

        acc_ctx = []
        acc_no_ctx = []
        for i in range(pc.num_levels):
            batch = interpolated[i].to(device)
            lab = interpolated_labels[i].to(device)
            acc_ctx.append(evaluate_discriminator(discriminator, batch, lab, context_associated))
            acc_no_ctx.append(evaluate_discriminator(discriminator, batch, lab, context_none))

        acc_ctx = np.array(acc_ctx)
        acc_no_ctx = np.array(acc_no_ctx)

        # Compute proportions
        x_ctx, prop_ctx = compute_proportions_from_accuracies(acc_ctx, pc.num_levels)
        x_no_ctx, prop_no_ctx = compute_proportions_from_accuracies(acc_no_ctx, pc.num_levels)

        # Fit psychometric curves
        fit_ctx = fit_psychometric_curve(x_ctx, prop_ctx)
        fit_no_ctx = fit_psychometric_curve(x_no_ctx, prop_no_ctx)

        # Bootstrap CIs
        ci_ctx_low, ci_ctx_high, _ = bootstrap_pse_ci(x_ctx, prop_ctx, n_bootstrap=pc.n_bootstrap)
        ci_no_low, ci_no_high, _ = bootstrap_pse_ci(x_no_ctx, prop_no_ctx, n_bootstrap=pc.n_bootstrap)

        # PSE shift
        shift_result = compute_pse_shift(
            fit_ctx.pse, fit_no_ctx.pse,
            (ci_ctx_low, ci_ctx_high), (ci_no_low, ci_no_high)
        )

        results[(digit_a, digit_b)] = {
            'fit_ctx': fit_ctx,
            'fit_no_ctx': fit_no_ctx,
            'x_ctx': x_ctx,
            'prop_ctx': prop_ctx,
            'x_no_ctx': x_no_ctx,
            'prop_no_ctx': prop_no_ctx,
            'shift': shift_result,
            'discriminator': discriminator,
        }

        print(f"  PSE (ctx):    {fit_ctx.pse:.4f} [{ci_ctx_low:.4f}, {ci_ctx_high:.4f}]")
        print(f"  PSE (no ctx): {fit_no_ctx.pse:.4f} [{ci_no_low:.4f}, {ci_no_high:.4f}]")
        print(f"  Shift: {shift_result['shift']:.4f} ({shift_result['direction']}), significant={shift_result['significant']}")

    return results


# Needed for torch in run_full_psychometric_pipeline
import torch
