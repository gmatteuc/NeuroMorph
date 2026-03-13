import numpy as np
import matplotlib.pyplot as plt

from .style import apply_neuromorph_style, COLOR_CONTEXT, COLOR_NO_CONTEXT, COLOR_REFERENCE


def plot_psychometric_curve(x_ctx, prop_ctx, x_no_ctx, prop_no_ctx,
                            fit_ctx=None, fit_no_ctx=None,
                            ci_ctx=None, ci_no_ctx=None,
                            digit_a=3, digit_b=4, ax=None):
    apply_neuromorph_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Raw data points
    ax.plot(x_ctx, prop_ctx, 'o', color=COLOR_CONTEXT, alpha=0.5, markersize=4)
    ax.plot(x_no_ctx, prop_no_ctx, 'o', color=COLOR_NO_CONTEXT, alpha=0.5, markersize=4)

    # Fitted curves
    if fit_ctx is not None:
        ax.plot(fit_ctx.x_fit, fit_ctx.y_fit, '-', color=COLOR_CONTEXT, linewidth=2,
                label=f'With context (PSE={fit_ctx.pse:.3f})')
        ax.axvline(fit_ctx.pse, color=COLOR_CONTEXT, linestyle=':', alpha=0.7)
    else:
        ax.plot(x_ctx, prop_ctx, '-', color=COLOR_CONTEXT, linewidth=2, label='With context')

    if fit_no_ctx is not None:
        ax.plot(fit_no_ctx.x_fit, fit_no_ctx.y_fit, '--', color=COLOR_NO_CONTEXT, linewidth=2,
                label=f'No context (PSE={fit_no_ctx.pse:.3f})')
        ax.axvline(fit_no_ctx.pse, color=COLOR_NO_CONTEXT, linestyle=':', alpha=0.7)
    else:
        ax.plot(x_no_ctx, prop_no_ctx, '--', color=COLOR_NO_CONTEXT, linewidth=2, label='No context')

    # CI shading
    if ci_ctx is not None:
        ax.axvspan(ci_ctx[0], ci_ctx[1], alpha=0.15, color=COLOR_CONTEXT)
    if ci_no_ctx is not None:
        ax.axvspan(ci_no_ctx[0], ci_no_ctx[1], alpha=0.15, color=COLOR_NO_CONTEXT)

    ax.axhline(0.5, color=COLOR_REFERENCE, linestyle='--', linewidth=1)
    ax.axvline(0.5, color=COLOR_REFERENCE, linestyle='--', linewidth=1)

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f'Clear "{digit_b}"', "Ambiguous", f'Clear "{digit_a}"'])
    ax.set_xlabel('Stimulus clarity (λ)')
    ax.set_ylabel(f'Proportion of choices "{digit_a}"')
    ax.set_title(f'Psychometric curves: "{digit_a}" vs "{digit_b}"')
    ax.legend()

    return ax


def plot_pse_shift_summary(results, ax=None):
    apply_neuromorph_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    pairs = list(results.keys())
    shifts = [results[p]['shift']['shift'] for p in pairs]

    # Error bars from bootstrap CIs
    ci_widths = []
    for p in pairs:
        s = results[p]['shift']
        ci_ctx = s['ci_ctx']
        ci_no_ctx = s['ci_no_ctx']
        # Approximate error as half-width of each CI added in quadrature
        err = np.sqrt(((ci_ctx[1] - ci_ctx[0]) / 2) ** 2 + ((ci_no_ctx[1] - ci_no_ctx[0]) / 2) ** 2)
        ci_widths.append(err)

    x_pos = np.arange(len(pairs))
    colors = [COLOR_CONTEXT if s > 0 else COLOR_NO_CONTEXT for s in shifts]
    bars = ax.bar(x_pos, shifts, yerr=ci_widths, color=colors, capsize=5, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{a} vs {b}' for a, b in pairs])
    ax.set_ylabel('PSE Shift (context - no context)')
    ax.set_title('PSE Shift Summary Across Digit Pairs')
    ax.axhline(0, color=COLOR_REFERENCE, linestyle='--', linewidth=1)

    # Mark significance
    for i, p in enumerate(pairs):
        if results[p]['shift']['significant']:
            ax.text(i, shifts[i] + ci_widths[i] + 0.005, '*', ha='center', fontsize=16, color=COLOR_CONTEXT)

    return ax
