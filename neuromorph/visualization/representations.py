import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

from .style import apply_neuromorph_style, CMAP


def plot_tsne_and_rdm(activation, labels, title, ax_tsne, ax_rdm, num_samples_per_class=10):
    selected_indices = np.concatenate([
        np.where(labels == i)[0][:num_samples_per_class] for i in range(10)
    ])
    selected_activation = activation[selected_indices]
    selected_labels = labels[selected_indices]

    tsne = TSNE(n_components=2, random_state=0)
    reduced_data = tsne.fit_transform(selected_activation.reshape(selected_activation.shape[0], -1))

    ax_tsne.set_title(f'{title} t-SNE')
    for i, (x, y) in enumerate(reduced_data):
        ax_tsne.text(x, y, str(selected_labels[i]),
                     color=plt.cm.plasma(selected_labels[i] / 10),
                     fontdict={'weight': 'bold', 'size': 9})
    scatter = ax_tsne.scatter(reduced_data[:, 0], reduced_data[:, 1],
                               c=selected_labels, cmap=CMAP, alpha=0.5)
    ax_tsne.figure.colorbar(scatter, ax=ax_tsne, ticks=range(10))

    sorted_indices = np.argsort(selected_labels)
    sorted_activation = selected_activation[sorted_indices]
    sorted_labels = selected_labels[sorted_indices]
    rdm = squareform(pdist(sorted_activation.reshape(sorted_activation.shape[0], -1), 'euclidean'))

    within_distances = []
    across_distances = []
    for i in range(10):
        indices = np.where(sorted_labels == i)[0]
        other_indices = np.where(sorted_labels != i)[0]
        if len(indices) > 1:
            within_distances.append(np.mean(rdm[np.ix_(indices, indices)]))
        if len(indices) > 0 and len(other_indices) > 0:
            across_distances.append(np.mean(rdm[np.ix_(indices, other_indices)]))
    avg_within = np.mean(within_distances)
    avg_across = np.mean(across_distances)

    rdm_plot = ax_rdm.imshow(rdm, cmap=CMAP, interpolation='nearest')
    ax_rdm.set_title(f'{title} RDM\nWithin: {avg_within:.2f}, Across: {avg_across:.2f}')
    ax_rdm.figure.colorbar(rdm_plot, ax=ax_rdm)

    class_size = num_samples_per_class
    for i in range(10):
        ax_rdm.axhline(i * class_size - 0.5, color='white', linewidth=1.5)
        ax_rdm.axvline(i * class_size - 0.5, color='white', linewidth=1.5)
        ax_rdm.text(-2.5, i * class_size + class_size / 2 - 0.5,
                    f'"{i}"', va='center', ha='right', fontsize=10, color='black')
        ax_rdm.text(i * class_size + class_size / 2 - 0.5, sorted_labels.size + 2,
                    f'"{i}"', va='center', ha='center', fontsize=10, color='black')
    ax_rdm.set_xticks([])
    ax_rdm.set_yticks([])


def plot_tsne_flow_field(tsne_no_ctx, tsne_ctx, labels, ax=None):
    apply_neuromorph_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title("Shift in t-SNE space (No Context vs Context)")
    for i, (x_nc, y_nc) in enumerate(tsne_no_ctx):
        x_wc, y_wc = tsne_ctx[i]
        ax.arrow(x_nc, y_nc, x_wc - x_nc, y_wc - y_nc,
                 color=plt.cm.plasma(labels[i] / 10), head_width=0.2, alpha=0.5)
        ax.scatter(x_nc, y_nc, color=plt.cm.plasma(labels[i] / 10), s=20, alpha=0.6)

    combined = np.concatenate([tsne_no_ctx, tsne_ctx], axis=0)
    ax.set_xlim(combined[:, 0].min(), combined[:, 0].max())
    ax.set_ylim(combined[:, 1].min(), combined[:, 1].max())


def plot_rdm_comparison(rdm, labels, samples_per_class, target_digit=3, ax_rdm=None, ax_bar=None):
    apply_neuromorph_style()
    if ax_rdm is None or ax_bar is None:
        fig, (ax_rdm, ax_bar) = plt.subplots(1, 2, figsize=(18, 10))

    rdm_plot = ax_rdm.imshow(rdm, cmap=CMAP, interpolation='nearest')
    min_val = np.min(np.ma.masked_where(np.eye(len(labels), dtype=bool), rdm))
    max_val = np.max(rdm)
    rdm_plot.set_clim(min_val, max_val)

    cbar = ax_rdm.figure.colorbar(rdm_plot, ax=ax_rdm, shrink=0.75)
    ax_rdm.set_title("Combined RDM (cosine)")

    for i in range(20):
        ax_rdm.axhline(i * samples_per_class - 0.5, color='white', linewidth=1.5)
        ax_rdm.axvline(i * samples_per_class - 0.5, color='white', linewidth=1.5)
        label_text = str(i % 10) + ("c" if i >= 10 else "")
        ax_rdm.text(-5, i * samples_per_class + samples_per_class / 2 - 0.5,
                    label_text, color="white", fontsize=10, ha='right')
        ax_rdm.text(i * samples_per_class + samples_per_class / 2 - 0.5, len(labels) + 5,
                    label_text, color="white", fontsize=10, ha='center')
    ax_rdm.set_xticks([])
    ax_rdm.set_yticks([])

    # Average dissimilarity from target digit
    avg_dissimilarities = [
        rdm[labels == target_digit, labels == i].mean()
        for i in range(10)
    ] + [
        rdm[labels == target_digit, labels == i + 10].mean()
        for i in range(10)
    ]

    ax_bar.bar(range(20), avg_dissimilarities, color=plt.cm.plasma(np.arange(20) / 20))
    ax_bar.set_xticks(range(20))
    ax_bar.set_xticklabels([str(i) for i in range(10)] + [f"{i}c" for i in range(10)])
    ax_bar.set_title(f"Average dissimilarity from '{target_digit}'")
    ax_bar.set_xlabel("Category")
    ax_bar.set_ylabel("Average dissimilarity")
    ax_bar.set_box_aspect(1)
