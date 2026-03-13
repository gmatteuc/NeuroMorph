import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CMAP = 'plasma'
CMAP_IMAGE = 'gray'
COLOR_CONTEXT = '#FFFF00'
COLOR_NO_CONTEXT = '#FFFFFF'
COLOR_REFERENCE = '#808080'
COLOR_NETWORK = '#FCA636'      # Plasma amber — network fitted curves
COLOR_BAYESIAN = '#9C179E'     # Plasma purple — Bayesian optimal curves
COLOR_DIGIT_A = '#ED7953'      # Plasma orange — digit A category
COLOR_DIGIT_B = '#CC4678'      # Plasma magenta — digit B category
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'plasma_div', ['#9C179E', '#2d2d2d', '#FCA636']
)  # Plasma-based diverging: purple (neg) → dark (zero) → amber (pos)
OUTPUT_DIR = './output'


def apply_neuromorph_style():
    plt.style.use('dark_background')


def save_figure(fig, name, dpi=300, output_dir=None):
    """Save figure to output directory as high-res PNG."""
    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    if not name.endswith('.png'):
        name += '.png'
    path = os.path.join(out, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {path}")
