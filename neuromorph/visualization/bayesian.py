import numpy as np
import matplotlib.pyplot as plt

from .style import (
    apply_neuromorph_style, CMAP, COLOR_CONTEXT, COLOR_NO_CONTEXT,
    COLOR_NETWORK, COLOR_BAYESIAN, COLOR_DIGIT_A, COLOR_DIGIT_B,
    CMAP_DIVERGING,
)


def plot_category_histogram(histograms, labels=None, ax=None):
    apply_neuromorph_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if labels is None:
        labels = list(histograms.keys())

    n_conditions = len(labels)
    x = np.arange(10)
    bar_width = 0.8 / n_conditions

    for i, label in enumerate(labels):
        if label == 'no_context':
            color = COLOR_NO_CONTEXT
        else:
            color = COLOR_CONTEXT  # yellow for any context condition
        ax.bar(x + i * bar_width, histograms[label], bar_width,
               label=label, color=color, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + bar_width * (n_conditions - 1) / 2)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel('Classified Digit')
    ax.set_ylabel('Proportion')
    ax.set_title('Category Distribution by Condition')
    ax.legend()

    return ax


def plot_sample_images(row_labels, row_data, title=None):
    """Grid of sample images with labeled rows."""
    apply_neuromorph_style()
    n_rows = len(row_labels)
    n_cols = row_data[0].size(0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.2 * n_rows))
    for r, (label, imgs) in enumerate(zip(row_labels, row_data)):
        for c in range(n_cols):
            axes[r, c].imshow(
                imgs[c].squeeze().clamp(-1, 1).mul(0.5).add(0.5).numpy(),
                cmap='gray'
            )
            axes[r, c].axis('off')
        axes[r, 0].text(
            -0.1, 0.5, label, transform=axes[r, 0].transAxes,
            fontsize=10, fontweight='bold', va='center', ha='right', color='white'
        )

    if title:
        plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_basin_analysis(prop_a_ctx, prop_b_ctx, prop_other_ctx,
                        prop_a_nc, prop_b_nc, prop_other_nc,
                        digit_order, digit_a, digit_b, noise_level=None):
    """Stacked bar chart of classification proportions, ordered by latent similarity."""
    apply_neuromorph_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(10)
    w = 0.6
    xlabels = [str(d) for d in digit_order]

    # Left: with context
    axes[0].bar(x, prop_a_ctx[digit_order], w,
                label=f'→ digit {digit_a}', color=COLOR_DIGIT_A)
    axes[0].bar(x, prop_b_ctx[digit_order], w,
                bottom=prop_a_ctx[digit_order],
                label=f'→ digit {digit_b}', color=COLOR_DIGIT_B)
    axes[0].bar(x, prop_other_ctx[digit_order], w,
                bottom=prop_a_ctx[digit_order] + prop_b_ctx[digit_order],
                label='→ other', color='gray', alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels)
    axes[0].set_xlabel(f'Original Digit (ordered by latent similarity to {digit_a} vs {digit_b})')
    axes[0].set_ylabel('Proportion')
    axes[0].set_title('With Context')
    axes[0].legend()

    # Right: without context
    axes[1].bar(x, prop_a_nc[digit_order], w,
                label=f'→ digit {digit_a}', color=COLOR_DIGIT_A)
    axes[1].bar(x, prop_b_nc[digit_order], w,
                bottom=prop_a_nc[digit_order],
                label=f'→ digit {digit_b}', color=COLOR_DIGIT_B)
    axes[1].bar(x, prop_other_nc[digit_order], w,
                bottom=prop_a_nc[digit_order] + prop_b_nc[digit_order],
                label='→ other', color='gray', alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_xlabel('Original Digit (same ordering)')
    axes[1].set_ylabel('Proportion')
    axes[1].set_title('Without Context (baseline)')
    axes[1].legend()

    noise_str = f' (noise_level={noise_level})' if noise_level is not None else ''
    plt.suptitle(
        f'Basin of Attraction — digits ordered by latent proximity to {digit_a} vs {digit_b}{noise_str}',
        fontsize=14
    )
    plt.tight_layout()
    return fig


def plot_bayesian_sigmoid(bin_centers, p_a_observed, bin_counts,
                          fitted_curve, fitted_params,
                          bayesian_x, bayesian_p_a,
                          digit_a, digit_b,
                          context_proj=None, noise_std=None, beta_opt=None,
                          min_display_count=10):
    """Per-image sigmoid with fitted curve and Bayesian optimal prediction."""
    apply_neuromorph_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Observed data (scatter with size proportional to count)
    valid = (~np.isnan(p_a_observed)) & (bin_counts >= min_display_count)
    sizes = np.clip(bin_counts[valid] / bin_counts[valid].max() * 200, 20, 200)
    ax.scatter(bin_centers[valid], p_a_observed[valid], s=sizes,
               color='white', edgecolor='white', alpha=0.8, zorder=3,
               label='Observed P(→{})'.format(digit_a))

    # Fitted sigmoid
    if fitted_curve is not None:
        ax.plot(bin_centers, fitted_curve, '--', color=COLOR_NETWORK, linewidth=2,
                label='Fitted sigmoid', zorder=2)
        if fitted_params is not None:
            alpha, beta, gamma, delta = fitted_params
            ax.axvline(alpha, color=COLOR_NETWORK, linestyle=':', alpha=0.5)

    # Bayesian optimal
    if bayesian_x is not None and bayesian_p_a is not None:
        ax.plot(bayesian_x, bayesian_p_a, '-', color=COLOR_BAYESIAN, linewidth=2.5,
                label='Bayesian optimal', zorder=2)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.4, label='Midpoint (equidistant)')

    ax.set_xlabel(f'Latent projection onto {digit_a}←→{digit_b} axis\n'
                  f'(← more {digit_b}-like | more {digit_a}-like →)')
    ax.set_ylabel(f'P(→ {digit_a} | classified as {digit_a} or {digit_b})')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')

    # Annotation with parameters
    info_lines = []
    if fitted_params is not None:
        alpha, beta, gamma, delta = fitted_params
        info_lines.append(f'Fitted: midpoint={alpha:.2f}, slope={beta:.2f}')
        info_lines.append(f'  asymptotes=[{gamma:.2f}, {delta:.2f}]')
    if beta_opt is not None:
        info_lines.append(f'Bayesian: slope={beta_opt:.2f}')
    if context_proj is not None:
        info_lines.append(f'Context shift (on axis): {context_proj:.2f}')
    if noise_std is not None:
        info_lines.append(f'Within-class noise σ: {noise_std:.2f}')
    if info_lines:
        ax.text(0.02, 0.98, '\n'.join(info_lines), transform=ax.transAxes,
                fontsize=9, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    ax.set_title(f'Bayesian Optimality: P(→{digit_a}) vs latent position')
    plt.tight_layout()
    return fig


def plot_centroid_rdm(rdm, linkage_matrix, metric='euclidean'):
    """Heatmap of pairwise centroid distances with dendrogram."""
    from scipy.cluster.hierarchy import dendrogram
    apply_neuromorph_style()

    n = rdm.shape[0]
    leaf_max = n * 10  # scipy places leaves at 5, 15, …, 95

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.10, 1], height_ratios=[0.10, 1],
                          wspace=0.0, hspace=0.0)

    # Top dendrogram
    ax_dendro_top = fig.add_subplot(gs[0, 1])
    dn = dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True,
                    color_threshold=0, above_threshold_color='white')
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_yticks([])
    ax_dendro_top.spines[:].set_visible(False)
    ax_dendro_top.set_xlim(0, leaf_max)

    order = dn['leaves']

    # Left dendrogram
    ax_dendro_left = fig.add_subplot(gs[1, 0])
    dendrogram(linkage_matrix, ax=ax_dendro_left, no_labels=True,
               orientation='left', color_threshold=0, above_threshold_color='white')
    ax_dendro_left.set_xticks([])
    ax_dendro_left.set_yticks([])
    ax_dendro_left.spines[:].set_visible(False)
    ax_dendro_left.set_ylim(0, leaf_max)
    ax_dendro_left.invert_yaxis()

    # Hide top-left corner
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis('off')

    # Heatmap (reordered)
    ax_heat = fig.add_subplot(gs[1, 1])
    rdm_ordered = rdm[np.ix_(order, order)]
    im = ax_heat.imshow(rdm_ordered, cmap='plasma', aspect='equal')

    # Tick labels on both sides of the matrix
    digit_labels = [str(d) for d in order]
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(digit_labels, fontsize=12, fontweight='bold')
    ax_heat.set_yticklabels(digit_labels, fontsize=12, fontweight='bold')
    ax_heat.tick_params(axis='x', bottom=True, top=True, labelbottom=True, labeltop=True)
    ax_heat.tick_params(axis='y', left=True, right=True, labelleft=True, labelright=True)

    # Annotate cells — adaptive text color for readability
    for i in range(n):
        for j in range(n):
            val = rdm_ordered[i, j]
            color = 'black' if val > rdm.max() * 0.6 else 'white'
            ax_heat.text(j, i, f'{val:.1f}', ha='center', va='center',
                         fontsize=8, color=color)

    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.set_label(f'{metric.capitalize()} distance')

    # Title on the figure (above everything, no overlap with dendrogram)
    fig.suptitle('Latent Centroid Distances (hierarchically clustered)',
                 fontsize=14, y=0.98)

    return fig, order


def plot_all_pairs_summary(results_df):
    """Summary plots for all 45 digit pair analyses: fitted sigmoid parameters."""
    apply_neuromorph_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Fitted slope heatmap
    slope_matrix = np.full((10, 10), np.nan)
    for _, row in results_df.iterrows():
        a, b = int(row['digit_a']), int(row['digit_b'])
        val = row['fitted_slope']
        if val is not None:
            slope_matrix[a, b] = val
            slope_matrix[b, a] = val

    im1 = axes[0].imshow(slope_matrix, cmap='plasma')
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    axes[0].set_title('Fitted Sigmoid Slope')
    med = np.nanmedian(slope_matrix)
    for i in range(10):
        for j in range(10):
            if not np.isnan(slope_matrix[i, j]):
                color = 'black' if slope_matrix[i, j] > med else 'white'
                axes[0].text(j, i, f'{slope_matrix[i, j]:.1f}', ha='center',
                             va='center', fontsize=7, color=color)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 2. Midpoint shift heatmap (signed)
    midpoint_matrix = np.full((10, 10), np.nan)
    for _, row in results_df.iterrows():
        a, b = int(row['digit_a']), int(row['digit_b'])
        mp = row['fitted_midpoint']
        if mp is not None:
            midpoint_matrix[a, b] = mp
            midpoint_matrix[b, a] = -mp

    vmax = np.nanmax(np.abs(midpoint_matrix))
    im2 = axes[1].imshow(midpoint_matrix, cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    axes[1].set_title('Sigmoid Midpoint Shift\n(amber = row-digit bias, purple = col-digit bias)')
    for i in range(10):
        for j in range(10):
            if not np.isnan(midpoint_matrix[i, j]):
                axes[1].text(j, i, f'{midpoint_matrix[i, j]:.1f}', ha='center',
                             va='center', fontsize=7, color='white')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle('All-Pairs Fitted Sigmoid Parameters (45 digit pairs)', fontsize=16, y=1.01)
    plt.tight_layout()
    return fig


def plot_all_pairs_sigmoid_mosaic(curves_data, results_df):
    """9x5 mosaic of sigmoid curves for all 45 pairs.

    curves_data: dict from np.load of the .npz file
    results_df: DataFrame with scalar metrics
    """
    from itertools import combinations
    apply_neuromorph_style()

    pairs = list(combinations(range(10), 2))
    fig, axes = plt.subplots(9, 5, figsize=(25, 36))

    for idx, (da, db) in enumerate(pairs):
        row, col = divmod(idx, 5)
        ax = axes[row, col]
        key = f'{da}v{db}'

        bin_centers = curves_data[f'{key}_bin_centers']
        p_a = curves_data[f'{key}_p_a_observed']
        bin_counts = curves_data[f'{key}_bin_counts']
        fitted_curve = curves_data[f'{key}_fitted_curve']
        bayes_x = curves_data[f'{key}_bayesian_x']
        bayes_p = curves_data[f'{key}_bayesian_p_a']

        # Observed data (scatter)
        valid = (~np.isnan(p_a)) & (bin_counts >= 10)
        if valid.any():
            sizes = np.clip(bin_counts[valid] / max(bin_counts[valid].max(), 1) * 80, 10, 80)
            ax.scatter(bin_centers[valid], p_a[valid], s=sizes,
                       color='white', edgecolor='white', alpha=0.7, zorder=3)

        # Fitted sigmoid
        if not np.all(np.isnan(fitted_curve)):
            ax.plot(bin_centers, fitted_curve, '--', color=COLOR_NETWORK, linewidth=1.5, zorder=2)

        # Non-parametric Bayesian optimal
        ax.plot(bayes_x, bayes_p, '-', color=COLOR_BAYESIAN, linewidth=1.5, zorder=2)

        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(bin_centers[0] - 0.2, bin_centers[-1] + 0.2)

        # Get slope ratio for title
        pair_row = results_df[(results_df['digit_a'] == da) & (results_df['digit_b'] == db)]
        if len(pair_row) > 0 and pair_row.iloc[0]['np_slope_ratio'] is not None:
            sr = pair_row.iloc[0]['np_slope_ratio']
            ax.set_title(f'{da} vs {db}  (r={sr:.2f})', fontsize=9, pad=2)
        else:
            ax.set_title(f'{da} vs {db}', fontsize=9, pad=2)

        ax.tick_params(labelsize=6)
        if col > 0:
            ax.set_yticklabels([])
        if row < 8:
            ax.set_xticklabels([])

    # Legend in first subplot
    axes[0, 0].plot([], [], '--', color=COLOR_NETWORK, linewidth=1.5, label='Network (fitted)')
    axes[0, 0].plot([], [], '-', color=COLOR_BAYESIAN, linewidth=1.5, label='Bayesian optimal')
    axes[0, 0].legend(fontsize=7, loc='lower right')

    fig.supxlabel('Latent projection onto decision axis', fontsize=12)
    fig.supylabel('P(classify as digit a)', fontsize=12)
    plt.suptitle('All 45 Digit Pairs: Network vs Non-Parametric Bayesian Optimal',
                 fontsize=16, y=1.005)
    plt.tight_layout()
    return fig


def plot_np_bayesian_summary(results_df):
    """Summary plots for non-parametric Bayesian analysis."""
    from scipy.stats import linregress
    apply_neuromorph_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. NP slope ratio heatmap
    sr_matrix = np.full((10, 10), np.nan)
    for _, row in results_df.iterrows():
        a, b = int(row['digit_a']), int(row['digit_b'])
        val = row['np_slope_ratio']
        if val is not None:
            sr_matrix[a, b] = val
            sr_matrix[b, a] = val

    im1 = axes[0, 0].imshow(sr_matrix, cmap='plasma', vmin=0.5, vmax=2.5)
    axes[0, 0].set_xticks(range(10))
    axes[0, 0].set_yticks(range(10))
    axes[0, 0].set_title('NP Slope Ratio (network / Bayesian optimal)')
    for i in range(10):
        for j in range(10):
            if not np.isnan(sr_matrix[i, j]):
                color = 'black' if sr_matrix[i, j] > 1.5 else 'white'
                axes[0, 0].text(j, i, f'{sr_matrix[i, j]:.2f}', ha='center',
                                va='center', fontsize=7, color=color)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 2. Midpoint calibration scatter + regression line
    valid = results_df.dropna(subset=['fitted_midpoint', 'np_bayesian_midpoint'])
    x_vals = valid['np_bayesian_midpoint'].values.astype(float)
    y_vals = valid['fitted_midpoint'].values.astype(float)

    axes[0, 1].scatter(x_vals, y_vals,
                       s=80, c=valid['np_slope_ratio'], cmap='plasma',
                       edgecolor='white', linewidth=0.5, vmin=0.5, vmax=2.5)

    # Identity line
    lims = [min(x_vals.min(), y_vals.min()) - 0.3,
            max(x_vals.max(), y_vals.max()) + 0.3]
    axes[0, 1].plot(lims, lims, '--', color='gray', alpha=0.4, label='Identity')

    # Regression line with confidence band
    reg = linregress(x_vals, y_vals)
    x_reg = np.linspace(lims[0], lims[1], 100)
    y_reg = reg.slope * x_reg + reg.intercept
    n = len(x_vals)
    x_mean = x_vals.mean()
    ss_x = ((x_vals - x_mean) ** 2).sum()
    resid_se = np.sqrt(np.sum((y_vals - (reg.slope * x_vals + reg.intercept)) ** 2) / (n - 2))
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, n - 2)
    se_line = resid_se * np.sqrt(1.0 / n + (x_reg - x_mean) ** 2 / ss_x)
    axes[0, 1].plot(x_reg, y_reg, '-', color=COLOR_NETWORK, linewidth=2,
                    label=f'Regression (r={reg.rvalue:.2f}, p={reg.pvalue:.1e})')
    axes[0, 1].fill_between(x_reg, y_reg - t_crit * se_line, y_reg + t_crit * se_line,
                            color=COLOR_NETWORK, alpha=0.15)

    for _, row in valid.iterrows():
        axes[0, 1].annotate(f"{int(row['digit_a'])}-{int(row['digit_b'])}",
                            (row['np_bayesian_midpoint'], row['fitted_midpoint']),
                            fontsize=6, color='white', ha='center', va='bottom')
    axes[0, 1].set_xlabel('Bayesian optimal midpoint')
    axes[0, 1].set_ylabel('Network fitted midpoint')
    axes[0, 1].set_title('Midpoint Calibration\n(color = NP slope ratio)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_xlim(lims)
    axes[0, 1].set_ylim(lims)

    # 3. Midpoint error heatmap — plasma-based diverging
    me_matrix = np.full((10, 10), np.nan)
    for _, row in results_df.iterrows():
        a, b = int(row['digit_a']), int(row['digit_b'])
        val = row['np_midpoint_error']
        if val is not None:
            me_matrix[a, b] = val
            me_matrix[b, a] = -val
    vmax_me = np.nanmax(np.abs(me_matrix))
    im3 = axes[1, 0].imshow(me_matrix, cmap=CMAP_DIVERGING, vmin=-vmax_me, vmax=vmax_me)
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_title('Midpoint Error (network - Bayesian)\n(amber = row-digit bias, purple = col-digit bias)')
    for i in range(10):
        for j in range(10):
            if not np.isnan(me_matrix[i, j]):
                axes[1, 0].text(j, i, f'{me_matrix[i, j]:.1f}', ha='center',
                                va='center', fontsize=7, color='white')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # 4. NP slope ratio vs centroid separation (scatter)
    valid_np = results_df.dropna(subset=['np_slope_ratio'])
    mean_np = valid_np['np_slope_ratio'].mean()
    std_np = valid_np['np_slope_ratio'].std()
    axes[1, 1].scatter(valid_np['delta_mu'], valid_np['np_slope_ratio'],
                       s=80, c=valid_np['balance_ratio'], cmap='plasma',
                       edgecolor='white', linewidth=0.5, vmin=0, vmax=1)
    for _, row in valid_np.iterrows():
        axes[1, 1].annotate(f"{int(row['digit_a'])}-{int(row['digit_b'])}",
                            (row['delta_mu'], row['np_slope_ratio']),
                            fontsize=6, color='white', ha='center', va='bottom')
    axes[1, 1].axhline(1.0, color='gray', linestyle=':', alpha=0.6, label='Bayesian optimal (1.0)')
    axes[1, 1].axhline(mean_np, color=COLOR_NETWORK, linestyle='--', linewidth=1.5,
                       label=f'Mean = {mean_np:.2f} +/- {std_np:.2f}')
    axes[1, 1].set_xlabel('Centroid Separation (delta_mu)')
    axes[1, 1].set_ylabel('NP Slope Ratio (network / Bayesian)')
    axes[1, 1].set_title('NP Slope Ratio vs Centroid Separation\n(color = balance ratio)')
    axes[1, 1].legend(fontsize=8)

    plt.suptitle('Non-Parametric Bayesian Analysis Summary (45 digit pairs)', fontsize=16, y=1.01)
    plt.tight_layout()
    return fig


def plot_average_sigmoid_v2(curves_data, results_df):
    """Average fitted vs non-parametric Bayesian optimal sigmoid, centered per pair."""
    from scipy.special import expit
    apply_neuromorph_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Common x-axis (centered per pair)
    x_common = np.linspace(-3, 3, 200)

    # Collect centered curves
    fitted_curves = []
    bayesian_curves = []

    for _, row in results_df.iterrows():
        da, db = int(row['digit_a']), int(row['digit_b'])
        key = f'{da}v{db}'

        # Fitted sigmoid, re-centered at its own midpoint
        if row['fitted_midpoint'] is not None and row['fitted_slope'] is not None:
            gamma = row['fitted_gamma'] if row['fitted_gamma'] is not None else 0.0
            delta = row['fitted_delta'] if row['fitted_delta'] is not None else 1.0
            curve = gamma + (delta - gamma) * expit(row['fitted_slope'] * x_common)
            fitted_curves.append(curve)

        # Bayesian curve, re-centered at its own midpoint
        if f'{key}_bayesian_x' in curves_data:
            bayes_x = curves_data[f'{key}_bayesian_x']
            bayes_p = curves_data[f'{key}_bayesian_p_a']
            bayes_mid = row['np_bayesian_midpoint'] if row['np_bayesian_midpoint'] is not None else 0.0
            centered_x = bayes_x - bayes_mid
            bayesian_curves.append(np.interp(x_common, centered_x, bayes_p))

    fitted_arr = np.array(fitted_curves)
    bayesian_arr = np.array(bayesian_curves)

    fitted_mean = fitted_arr.mean(axis=0)
    fitted_std = fitted_arr.std(axis=0)
    bayesian_mean = bayesian_arr.mean(axis=0)
    bayesian_std = bayesian_arr.std(axis=0)

    ax.fill_between(x_common, fitted_mean - fitted_std, fitted_mean + fitted_std,
                    color=COLOR_NETWORK, alpha=0.2)
    ax.plot(x_common, fitted_mean, '--', color=COLOR_NETWORK, linewidth=2.5,
            label='Network (mean fitted)')
    ax.fill_between(x_common, bayesian_mean - bayesian_std, bayesian_mean + bayesian_std,
                    color=COLOR_BAYESIAN, alpha=0.2)
    ax.plot(x_common, bayesian_mean, '-', color=COLOR_BAYESIAN, linewidth=2.5,
            label='Bayesian optimal (mean KDE)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Normalized latent position (centered per pair)')
    ax.set_ylabel('P(classify as digit a)')
    ax.set_title(f'Average Sigmoid: Network vs Bayesian Optimal\n(n={len(fitted_curves)} pairs, centered at midpoint)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    return fig
