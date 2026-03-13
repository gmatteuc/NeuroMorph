import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_balanced_rdm(activations_no_context, activations_context, labels, metric='cosine'):
    samples_per_class = min(np.sum(labels == i) for i in range(10))
    balanced_indices = np.concatenate([
        np.random.choice(np.where(labels == i)[0], samples_per_class, replace=False)
        for i in range(10)
    ])

    balanced_no_ctx = activations_no_context[balanced_indices]
    balanced_ctx = activations_context[balanced_indices]
    balanced_combined = np.concatenate([balanced_no_ctx, balanced_ctx], axis=0)
    balanced_labels = np.concatenate([labels[balanced_indices], labels[balanced_indices] + 10])

    rdm = squareform(pdist(balanced_combined, metric=metric))

    return rdm, balanced_labels, samples_per_class
