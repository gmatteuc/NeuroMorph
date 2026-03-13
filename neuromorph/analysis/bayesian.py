import torch
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.spatial.distance import pdist, squareform


def compute_digit_centroids(encoder, dataloader, config, device):
    """Compute mean latent vector per digit class (clean images, no context)."""
    from neuromorph.data import generate_context
    mc = config.model
    cc = config.context

    digit_latents = {d: [] for d in range(10)}
    encoder.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            no_ctx = generate_context(
                images.size(0), signal_level=None,
                noise_level=cc.noise_level, category_id=0,
                context_dim=mc.context_dim, device=device
            )
            latent = encoder(images, no_ctx).cpu()
            for d in range(10):
                mask = labels == d
                if mask.any():
                    digit_latents[d].append(latent[mask])

    centroids = {d: torch.cat(digit_latents[d]).mean(dim=0) for d in range(10)}
    return centroids


def compute_similarity_order(centroids, digit_a, digit_b):
    """Order digits by relative latent distance to digit_a vs digit_b.

    Returns:
        digit_order: list sorted from most b-like to most a-like
        scores: dict {digit: score} where positive = closer to a
    """
    centroid_a = centroids[digit_a]
    centroid_b = centroids[digit_b]

    scores = {}
    for d in range(10):
        dist_to_a = torch.norm(centroids[d] - centroid_a).item()
        dist_to_b = torch.norm(centroids[d] - centroid_b).item()
        scores[d] = dist_to_b - dist_to_a

    digit_order = sorted(range(10), key=lambda d: scores[d])
    return digit_order, scores


def run_corruption_test(autoencoder, classifier, dataloader, noise_level,
                        context_signal_level, context_id, config, device):
    """Feed noisy images through autoencoder with/without context, classify.

    Returns dict with orig_labels, preds_no_ctx, preds_ctx, noisy_latents,
    hist_no_ctx, hist_ctx, samples.
    """
    from neuromorph.data import generate_context, add_noise
    mc = config.model
    nc = config.noise
    bc = config.bimodal

    hist_no_ctx = np.zeros(10)
    hist_ctx = np.zeros(10)
    n_total = 0
    all_orig_labels, all_preds_no_ctx, all_preds_ctx = [], [], []
    all_noisy_latents = []
    samples = {}

    autoencoder.eval()
    classifier.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            bs = images.size(0)

            noisy = add_noise(images, noise_level=noise_level,
                              correlated=nc.correlated, uncorrelated=nc.uncorrelated,
                              kernel_size=nc.kernel_size, sigma=nc.sigma)

            no_ctx = generate_context(bs, signal_level=None,
                                      noise_level=bc.noise_level, category_id=0,
                                      context_dim=mc.context_dim, device=device)
            ctx = generate_context(bs, signal_level=context_signal_level,
                                   noise_level=bc.noise_level, category_id=context_id,
                                   context_dim=mc.context_dim, device=device)

            recon_none = autoencoder(noisy, no_ctx)
            recon_ctx = autoencoder(noisy, ctx)

            noisy_latent = autoencoder.encoder(noisy, no_ctx).cpu()
            all_noisy_latents.append(noisy_latent)

            preds_none = classifier.predict(recon_none).cpu()
            preds_ctx_batch = classifier.predict(recon_ctx).cpu()

            hist_no_ctx += np.bincount(preds_none.numpy(), minlength=10)
            hist_ctx += np.bincount(preds_ctx_batch.numpy(), minlength=10)
            n_total += bs

            all_orig_labels.append(labels)
            all_preds_no_ctx.append(preds_none)
            all_preds_ctx.append(preds_ctx_batch)

            if not samples:
                samples = {
                    'original': images[:10].cpu(),
                    'noisy': noisy[:10].cpu(),
                    'recon_no_ctx': recon_none[:10].cpu(),
                    'recon_ctx': recon_ctx[:10].cpu(),
                }

    return {
        'orig_labels': torch.cat(all_orig_labels).numpy(),
        'preds_no_ctx': torch.cat(all_preds_no_ctx).numpy(),
        'preds_ctx': torch.cat(all_preds_ctx).numpy(),
        'noisy_latents': torch.cat(all_noisy_latents),
        'hist_no_ctx': hist_no_ctx / n_total,
        'hist_ctx': hist_ctx / n_total,
        'samples': samples,
    }


def compute_basin_proportions(orig_labels, predictions, digit_a, digit_b):
    """Per-digit-class breakdown of classification proportions."""
    prop_a = np.zeros(10)
    prop_b = np.zeros(10)
    prop_other = np.zeros(10)

    for d in range(10):
        mask = orig_labels == d
        if mask.sum() == 0:
            continue
        n = mask.sum()
        preds = predictions[mask]
        prop_a[d] = (preds == digit_a).sum() / n
        prop_b[d] = (preds == digit_b).sum() / n
        prop_other[d] = 1 - prop_a[d] - prop_b[d]

    return prop_a, prop_b, prop_other


def compute_sigmoid_data(noisy_latents, preds_ctx, centroids, digit_a, digit_b,
                         n_bins=20):
    """Per-image sigmoid: P(→a | projection) as function of latent position on a-b axis.

    Projects each noisy image's latent onto the axis connecting centroid_b to centroid_a,
    bins by projection, computes P(→a) among images classified as a or b.

    Returns:
        bin_centers, p_a, bin_counts, projections
    """
    centroid_a = centroids[digit_a]
    centroid_b = centroids[digit_b]
    axis = centroid_a - centroid_b
    axis_norm = axis / torch.norm(axis)
    midpoint = (centroid_a + centroid_b) / 2

    projections = ((noisy_latents - midpoint) @ axis_norm).numpy()

    is_a = preds_ctx == digit_a
    is_b = preds_ctx == digit_b
    is_target = is_a | is_b

    lo = np.percentile(projections, 2)
    hi = np.percentile(projections, 98)
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    p_a = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (projections >= bin_edges[i]) & (projections < bin_edges[i + 1])
        target_in_bin = in_bin & is_target
        bin_counts[i] = target_in_bin.sum()
        if bin_counts[i] > 0:
            p_a[i] = (in_bin & is_a).sum() / target_in_bin.sum()

    return bin_centers, p_a, bin_counts, projections


def _sigmoid_func(x, alpha, beta, gamma, delta):
    """4-parameter sigmoid: gamma + (delta - gamma) * sigmoid(beta * (x - alpha))"""
    return gamma + (delta - gamma) * expit(beta * (x - alpha))


def fit_sigmoid(bin_centers, p_a, bin_counts, min_count=10):
    """Fit 4-parameter sigmoid to binned P(→a) data.

    Returns:
        params: (alpha, beta, gamma, delta) or None if fit fails
        fitted_curve: predicted values at bin_centers or None
    """
    valid = (~np.isnan(p_a)) & (bin_counts >= min_count)
    if valid.sum() < 4:
        return None, None

    x = bin_centers[valid]
    y = p_a[valid]

    try:
        params, _ = curve_fit(
            _sigmoid_func, x, y,
            p0=[0.0, 1.0, 0.05, 0.95],
            bounds=([-np.inf, 0, 0, 0], [np.inf, np.inf, 1, 1]),
            maxfev=5000
        )
        fitted_curve = _sigmoid_func(bin_centers, *params)
    except RuntimeError:
        return None, None

    return params, fitted_curve


def compute_bayesian_prediction(encoder, centroids, noisy_latents, orig_labels,
                                context_signal_level, context_id, config, device,
                                digit_a, digit_b):
    """Compute Bayesian-optimal sigmoid for comparison (parametric, legacy).

    Uses the latent geometry (centroid separation), within-class noise variance,
    and measured context shift to derive what an ideal Bayesian observer would predict.

    Returns:
        x_range: array for plotting
        bayesian_p_a: Bayesian optimal P(→a) at each x
        context_proj: context shift projected onto a-b axis
        noise_std: within-class noise std along decision axis
        beta_opt: optimal sigmoid slope
        delta_mu: centroid separation
    """
    from neuromorph.data import generate_context
    mc = config.model
    bc = config.bimodal

    centroid_a = centroids[digit_a]
    centroid_b = centroids[digit_b]
    axis = centroid_a - centroid_b
    axis_norm = axis / torch.norm(axis)
    midpoint = (centroid_a + centroid_b) / 2

    # Measure context shift along the decision axis
    with torch.no_grad():
        ctx = generate_context(1, signal_level=context_signal_level,
                               noise_level=bc.noise_level, category_id=context_id,
                               context_dim=mc.context_dim, device=device)
        no_ctx = generate_context(1, signal_level=None,
                                  noise_level=bc.noise_level, category_id=0,
                                  context_dim=mc.context_dim, device=device)
        shift_ctx = encoder.fc_context(ctx).cpu().squeeze()
        shift_no = encoder.fc_context(no_ctx).cpu().squeeze()

    context_shift = shift_ctx - shift_no
    context_proj = (context_shift @ axis_norm).item()

    # Within-class noise variance along the decision axis
    projections = ((noisy_latents - midpoint) @ axis_norm).numpy()
    proj_a = projections[orig_labels == digit_a]
    proj_b = projections[orig_labels == digit_b]
    noise_var = (proj_a.var() + proj_b.var()) / 2

    # Centroid separation along the axis (= full norm since axis IS the connecting direction)
    delta_mu = torch.norm(centroid_a - centroid_b).item()

    # Bayesian optimal: P(→a | d) = σ(β * (d + c))
    # where β = Δμ / σ² (from Gaussian log-likelihood ratio)
    beta_opt = delta_mu / noise_var if noise_var > 0 else 1.0

    lo = np.percentile(projections, 2)
    hi = np.percentile(projections, 98)
    x_range = np.linspace(lo, hi, 100)
    bayesian_p_a = expit(beta_opt * (x_range + context_proj))

    return x_range, bayesian_p_a, context_proj, noise_var ** 0.5, beta_opt, delta_mu


def compute_nonparametric_bayesian(noisy_latents, orig_labels, centroids,
                                   digit_a, digit_b, prior_a=0.5, n_points=200):
    """Non-parametric Bayesian optimal prediction from unperturbed visual geometry.

    Uses KDE to estimate class-conditional distributions P(d|a) and P(d|b)
    along the decision axis from unperturbed (no-context) latent projections,
    then combines with the training prior P(a|context) via Bayes' rule.

    This is a clean comparison: the Bayesian model knows only the visual geometry
    and the training statistics, NOT the network's learned context shift.

    Returns dict with:
        x_range, bayesian_p_a: the Bayesian optimal curve
        kde_a, kde_b: the fitted KDE objects
        delta_mu, sigma_a, sigma_b: geometric statistics
        bayesian_midpoint: where the Bayesian curve crosses 0.5
        bayesian_slope_at_midpoint: slope of the Bayesian curve at its midpoint
    """
    from scipy.stats import gaussian_kde

    centroid_a = centroids[digit_a]
    centroid_b = centroids[digit_b]
    axis = centroid_a - centroid_b
    axis_norm = axis / torch.norm(axis)
    midpoint = (centroid_a + centroid_b) / 2

    # Project all latents onto decision axis (unperturbed, no context)
    projections = ((noisy_latents - midpoint) @ axis_norm).numpy()

    # Class-conditional projections
    proj_a = projections[orig_labels == digit_a]
    proj_b = projections[orig_labels == digit_b]

    # KDE estimation of class-conditional densities
    kde_a = gaussian_kde(proj_a)
    kde_b = gaussian_kde(proj_b)

    # Evaluation grid
    lo = np.percentile(projections, 1)
    hi = np.percentile(projections, 99)
    x_range = np.linspace(lo, hi, n_points)

    # Bayesian posterior: P(a | d, ctx) = P(d|a) * P(a|ctx) / [P(d|a)*P(a|ctx) + P(d|b)*P(b|ctx)]
    prior_b = 1.0 - prior_a
    lik_a = kde_a(x_range)
    lik_b = kde_b(x_range)

    denom = lik_a * prior_a + lik_b * prior_b
    # Avoid division by zero in tails
    safe_denom = np.where(denom > 1e-30, denom, 1e-30)
    bayesian_p_a = (lik_a * prior_a) / safe_denom

    # Compute midpoint (where curve crosses 0.5)
    crossings = np.where(np.diff(np.sign(bayesian_p_a - 0.5)))[0]
    if len(crossings) > 0:
        # Linear interpolation at first crossing
        idx = crossings[0]
        x0, x1 = x_range[idx], x_range[idx + 1]
        y0, y1 = bayesian_p_a[idx], bayesian_p_a[idx + 1]
        bayesian_midpoint = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
    else:
        bayesian_midpoint = 0.0

    # Slope at midpoint (numerical derivative)
    dx = x_range[1] - x_range[0]
    gradient = np.gradient(bayesian_p_a, dx)
    if len(crossings) > 0:
        bayesian_slope = gradient[crossings[0]]
    else:
        bayesian_slope = gradient[n_points // 2]

    # Geometric statistics
    delta_mu = torch.norm(centroid_a - centroid_b).item()
    sigma_a = float(proj_a.std())
    sigma_b = float(proj_b.std())

    return {
        'x_range': x_range,
        'bayesian_p_a': bayesian_p_a,
        'likelihood_a': lik_a,
        'likelihood_b': lik_b,
        'delta_mu': delta_mu,
        'sigma_a': sigma_a,
        'sigma_b': sigma_b,
        'bayesian_midpoint': float(bayesian_midpoint),
        'bayesian_slope_at_midpoint': float(bayesian_slope),
    }


def compute_centroid_rdm(centroids, metric='euclidean'):
    """Compute pairwise distance matrix between digit centroids.

    Returns:
        rdm: 10x10 distance matrix
        linkage_matrix: scipy hierarchical clustering linkage
    """
    from scipy.cluster.hierarchy import linkage

    centroid_matrix = torch.stack([centroids[d] for d in range(10)]).numpy()
    dists = pdist(centroid_matrix, metric=metric)
    rdm = squareform(dists)
    linkage_matrix = linkage(dists, method='average')

    return rdm, linkage_matrix


def run_pair_analysis(autoencoder, classifier, encoder, dataloader,
                      centroids, digit_a, digit_b, context_id,
                      noise_level, context_signal_level, config, device,
                      prior_a=0.5):
    """Run full sigmoid + Bayesian analysis for a single digit pair.

    Returns dict with all metrics needed for the all-pairs summary,
    including both legacy (parametric) and non-parametric Bayesian metrics.
    """
    exp = run_corruption_test(
        autoencoder, classifier, dataloader, noise_level=noise_level,
        context_signal_level=context_signal_level,
        context_id=context_id, config=config, device=device
    )

    bin_centers, p_a, bin_counts, projections = compute_sigmoid_data(
        exp['noisy_latents'], exp['preds_ctx'], centroids,
        digit_a, digit_b, n_bins=25
    )

    fitted_params, fitted_curve = fit_sigmoid(bin_centers, p_a, bin_counts)

    # Legacy parametric Bayesian prediction
    bayes_x, bayes_p, context_proj, noise_std, beta_opt, delta_mu = \
        compute_bayesian_prediction(
            encoder, centroids, exp['noisy_latents'], exp['orig_labels'],
            context_signal_level=context_signal_level,
            context_id=context_id, config=config, device=device,
            digit_a=digit_a, digit_b=digit_b
        )

    # Non-parametric Bayesian prediction (from unperturbed geometry + training prior)
    np_bayes = compute_nonparametric_bayesian(
        exp['noisy_latents'], exp['orig_labels'], centroids,
        digit_a, digit_b, prior_a=prior_a
    )

    mass_a = exp['hist_ctx'][digit_a]
    mass_b = exp['hist_ctx'][digit_b]
    mass_total = mass_a + mass_b
    balance_ratio = min(mass_a, mass_b) / max(mass_a, mass_b) if max(mass_a, mass_b) > 0 else 0

    # Compute slope ratio against non-parametric Bayesian
    np_slope_ratio = None
    if fitted_params is not None and np_bayes['bayesian_slope_at_midpoint'] > 0:
        # Convert fitted sigmoid slope to slope in P-space at its own midpoint:
        # dP/dx at midpoint = beta * (delta - gamma) / 4
        gamma, delta = fitted_params[2], fitted_params[3]
        fitted_slope_at_midpoint = fitted_params[1] * (delta - gamma) / 4.0
        np_slope_ratio = fitted_slope_at_midpoint / np_bayes['bayesian_slope_at_midpoint']

    result = {
        'digit_a': digit_a,
        'digit_b': digit_b,
        'mass_a': float(mass_a),
        'mass_b': float(mass_b),
        'mass_total': float(mass_total),
        'balance_ratio': float(balance_ratio),
        'delta_mu': float(delta_mu),
        'noise_std': float(noise_std),
        'sigma_a': np_bayes['sigma_a'],
        'sigma_b': np_bayes['sigma_b'],
        'context_proj': float(context_proj),
        # Legacy parametric Bayesian
        'beta_opt': float(beta_opt),
        'fitted_midpoint': float(fitted_params[0]) if fitted_params is not None else None,
        'fitted_slope': float(fitted_params[1]) if fitted_params is not None else None,
        'fitted_gamma': float(fitted_params[2]) if fitted_params is not None else None,
        'fitted_delta': float(fitted_params[3]) if fitted_params is not None else None,
        'slope_ratio': float(fitted_params[1] / beta_opt) if fitted_params is not None and beta_opt > 0 else None,
        # Non-parametric Bayesian
        'np_bayesian_midpoint': np_bayes['bayesian_midpoint'],
        'np_bayesian_slope': np_bayes['bayesian_slope_at_midpoint'],
        'np_slope_ratio': float(np_slope_ratio) if np_slope_ratio is not None else None,
        'np_midpoint_error': float(fitted_params[0] - np_bayes['bayesian_midpoint']) if fitted_params is not None else None,
    }
    return result
