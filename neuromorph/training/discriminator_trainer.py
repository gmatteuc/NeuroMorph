import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from neuromorph.models import Discriminator
from neuromorph.data import generate_context, collect_all_digit_examples


def train_discriminator(association_ae, test_dataloader, config, device):
    dc = config.discriminator
    mc = config.model
    cc = config.context

    # Collect all examples for both digits
    images_a, labels_a = collect_all_digit_examples(test_dataloader, digit=dc.digit_a)
    labels_a[:] = 0
    images_b, labels_b = collect_all_digit_examples(test_dataloader, digit=dc.digit_b)
    labels_b[:] = 1

    all_images = torch.cat([images_a, images_b], dim=0)
    all_labels = torch.cat([labels_a, labels_b], dim=0)

    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_labels = train_labels.float().to(device)
    test_labels = test_labels.float().to(device)
    train_images = train_images.to(device)
    test_images = test_images.to(device)

    # Get latent dim
    sample_context = generate_context(
        1, signal_level=None, noise_level=cc.noise_level,
        category_id=cc.context_id, context_dim=mc.context_dim, device=device
    )
    latent_dim = association_ae.encoder(train_images[0:1], sample_context).view(1, -1).size(1)

    discriminator = Discriminator(association_ae.encoder, latent_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.logistic.parameters(), lr=dc.lr)

    for epoch in range(dc.epochs):
        discriminator.train()
        optimizer.zero_grad()
        context_train = generate_context(
            train_images.size(0), signal_level=None, noise_level=cc.noise_level,
            category_id=cc.context_id, context_dim=mc.context_dim, device=device
        )
        outputs = discriminator(train_images, context_train).squeeze()
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    train_acc = evaluate_discriminator(
        discriminator, train_images, train_labels,
        generate_context(train_images.size(0), signal_level=None, noise_level=cc.noise_level,
                         category_id=cc.context_id, context_dim=mc.context_dim, device=device)
    )
    test_context = generate_context(
        test_images.size(0), signal_level=None, noise_level=cc.noise_level,
        category_id=cc.context_id, context_dim=mc.context_dim, device=device
    )
    test_acc = evaluate_discriminator(discriminator, test_images, test_labels, test_context)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return discriminator


def evaluate_discriminator(model, images, labels, context):
    model.eval()
    with torch.no_grad():
        outputs = model(images, context).squeeze()
        predictions = torch.sigmoid(outputs) > 0.5
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    return accuracy
