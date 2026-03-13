import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from .style import apply_neuromorph_style, CMAP_IMAGE


def show_image(img, title="", ax=None):
    apply_neuromorph_style()
    img = img.cpu()
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    npimg = img.numpy()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap=CMAP_IMAGE)
    ax.set_title(title)
    ax.axis('off')
    return fig


def save_high_res_image(tensor, filename, nrow=5, figsize=(15, 15)):
    apply_neuromorph_style()
    grid_img = torchvision.utils.make_grid(tensor, nrow=nrow, padding=2, scale_each=True)
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_average_images(images, labels, reconstructions_with_context, reconstructions_no_context):
    apply_neuromorph_style()
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))

    for i in range(10):
        indices = np.where(labels.cpu().numpy() == i)[0]
        if len(indices) > 0:
            avg_original = images[indices].mean(dim=0).detach().cpu().numpy()
            avg_with_ctx = reconstructions_with_context[indices].mean(dim=0).detach().cpu().numpy()
            avg_no_ctx = reconstructions_no_context[indices].mean(dim=0).detach().cpu().numpy()

            axes[i, 0].imshow(avg_original.reshape(28, 28), cmap=CMAP_IMAGE)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Avg (Original)')

            axes[i, 1].imshow(avg_with_ctx.reshape(28, 28), cmap=CMAP_IMAGE)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('Avg (With Context)')

            axes[i, 2].imshow(avg_no_ctx.reshape(28, 28), cmap=CMAP_IMAGE)
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('Avg (No Context)')

            axes[i, 0].set_ylabel(f'Digit {i}', fontsize=12)

    plt.tight_layout()
    return fig


def visualize_noisy_images(original, noisy, num_images=5):
    apply_neuromorph_style()
    original = original[:num_images].cpu()
    noisy = noisy[:num_images].cpu()

    fig, axs = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        axs[0, i].imshow(original[i].squeeze().numpy(), cmap=CMAP_IMAGE)
        axs[0, i].axis('off')
        axs[0, i].set_title("Original")

        axs[1, i].imshow(noisy[i].squeeze().numpy(), cmap=CMAP_IMAGE)
        axs[1, i].axis('off')
        axs[1, i].set_title("Noisy")

    plt.suptitle("Comparison of Original and Noisy Images")
    return fig
