"""Backward compatibility shim — imports from neuromorph.data and neuromorph.visualization."""
from neuromorph.data.context import generate_context
from neuromorph.data.noise import add_noise
from neuromorph.visualization.images import visualize_noisy_images

__all__ = ['generate_context', 'add_noise', 'visualize_noisy_images']
