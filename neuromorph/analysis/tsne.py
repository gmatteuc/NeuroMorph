import numpy as np
from sklearn.manifold import TSNE


def compute_joint_tsne(activations_no_context, activations_context, labels, random_state=0):
    combined = np.concatenate([activations_no_context, activations_context], axis=0)
    tsne = TSNE(n_components=2, random_state=random_state)
    combined_tsne = tsne.fit_transform(combined)
    n = len(activations_no_context)
    return combined_tsne[:n], combined_tsne[n:]


def compute_distance_change(activations_no_context, activations_context, labels, target_digit=3):
    avg_no_ctx = activations_no_context[labels == target_digit].mean(axis=0)
    avg_ctx = activations_context[labels == target_digit].mean(axis=0)
    avg_ref = (avg_no_ctx + avg_ctx) / 2

    distance_diffs = []
    for i in range(10):
        d_ctx = np.linalg.norm(activations_context[labels == i] - avg_ref, axis=1).mean()
        d_no_ctx = np.linalg.norm(activations_no_context[labels == i] - avg_ref, axis=1).mean()
        distance_diffs.append(d_ctx - d_no_ctx)
    return distance_diffs
