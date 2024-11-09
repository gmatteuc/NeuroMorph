import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# unified context generation function for evaluation
def generate_context(batch_size, signal_level=None, noise_level=0.1, category_id=2, context_dim=10, device='cuda'):
    """
    Generates a context vector. If `high_value` is provided, it simulates an associated context by setting the specified 
    element (`category_id`) to the high value; otherwise, it generates a noisy, uninformative context.
    
    Parameters:
    - batch_size: Number of context vectors to generate.
    - signal_level: The value to set for the specified element (if None, generates only noise).
    - noise_level: The noise level applied to the context vectors.
    - category_id: The index in the context vector where the high value is set.
    - context_dim: Dimensionality of the context vector.
    - device: Device to place the tensor (e.g., 'cuda' or 'cpu').
    
    Returns:
    - A context tensor of shape (batch_size, context_dim).
    """
    context = noise_level * torch.randn(batch_size, context_dim).to(device)
    if signal_level is not None:
        context[:, category_id] = signal_level
    return context

# unified function to add noise to images
def add_noise(images, noise_level=1.0, kernel_size=5, sigma=2.0, correlated=False, uncorrelated=True):
    """
    Adds correlated, uncorrelated, or both types of noise to images.
    
    Parameters:
    - images: Tensor of shape (batch_size, channels, height, width).
    - noise_level: The amplitude of the noise.
    - kernel_size: Size of the Gaussian kernel for correlated noise (ignored if `correlated` is False).
    - sigma: Standard deviation of the Gaussian kernel (ignored if `correlated` is False).
    - correlated: If True, adds correlated noise by applying Gaussian smoothing.
    - uncorrelated: If True, adds uncorrelated (white) noise.
    
    Returns:
    - A tensor of images with added noise, clamped to the range [-1, 1].
    """
    
    # ensure images have a channel dimension (e.g., shape [batch_size, channels, height, width])
    if images.dim() == 3:
        images = images.unsqueeze(1)  # add channel dimension if missing

    noise = torch.zeros_like(images)

    if uncorrelated:
        noise += noise_level * torch.randn_like(images)

    if correlated:
        # generate correlated noise
        correlated_noise = torch.randn_like(images)

        # create a 2D Gaussian kernel
        grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gaussian_kernel = torch.exp(-0.5 * (grid**2) / sigma**2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize the kernel
        gaussian_kernel_2d = gaussian_kernel[:, None] * gaussian_kernel[None, :]
        gaussian_kernel_2d = gaussian_kernel_2d.to(images.device).unsqueeze(0).unsqueeze(0)

        # apply the Gaussian filter to each channel of the noise (batch_size, channels, H, W)
        correlated_noise = F.conv2d(correlated_noise, gaussian_kernel_2d, padding=kernel_size // 2, groups=1)

        # normalize the noise to have the requested amplitude
        correlated_noise = (2 * noise_level * correlated_noise / correlated_noise.std()) - noise_level

        noise += correlated_noise

    # add the noise to the original images and clamp the values
    noisy_images = images + noise
    return noisy_images.clamp(-1, 1)


def visualize_noisy_images(original, noisy, num_images=5):
    """
    Visualizes a comparison between original and noisy images.
    
    Parameters:
    - original: Tensor of original images.
    - noisy: Tensor of noisy images.
    - num_images: Number of images to visualize.

    Returns:
    - None
    """
    original = original[:num_images].cpu()  # move to CPU
    noisy = noisy[:num_images].cpu()  # move to CPU
    
    fig, axs = plt.subplots(2, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # Original images
        axs[0, i].imshow(original[i].squeeze().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title("Original")

        # Noisy images
        axs[1, i].imshow(noisy[i].squeeze().numpy(), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title("Noisy")
    
    plt.suptitle("Comparison of Original and Noisy Images")
    plt.show()