from .style import (
    apply_neuromorph_style, save_figure, CMAP, CMAP_IMAGE,
    COLOR_CONTEXT, COLOR_NO_CONTEXT, COLOR_REFERENCE,
    COLOR_NETWORK, COLOR_BAYESIAN, COLOR_DIGIT_A, COLOR_DIGIT_B,
    CMAP_DIVERGING,
)
from .images import show_image, save_high_res_image, plot_average_images, visualize_noisy_images
from .representations import plot_tsne_and_rdm, plot_tsne_flow_field, plot_rdm_comparison
from .psychometric import plot_psychometric_curve, plot_pse_shift_summary
from .bayesian import (
    plot_category_histogram, plot_sample_images, plot_basin_analysis,
    plot_bayesian_sigmoid, plot_centroid_rdm, plot_all_pairs_summary,
    plot_all_pairs_sigmoid_mosaic, plot_np_bayesian_summary, plot_average_sigmoid_v2,
)
