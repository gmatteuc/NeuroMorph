import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, capacity, latent_dims, context_dim, context_mode="linear"):
        super().__init__()
        self.context_mode = context_mode
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=capacity, out_channels=capacity * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=capacity * 2 * 7 * 7, out_features=latent_dims)

        if context_mode == "gated":
            self.fc_context = nn.Sequential(
                nn.Linear(context_dim + latent_dims, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dims),
            )
        elif context_mode == "mlp":
            self.fc_context = nn.Sequential(
                nn.Linear(context_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dims),
            )
        else:
            self.fc_context = nn.Linear(context_dim, latent_dims)

    def forward(self, image, context):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        visual_latent = self.fc1(x)

        if self.context_mode == "gated":
            context_projection = self.fc_context(torch.cat([context, visual_latent], dim=1))
        else:
            context_projection = self.fc_context(context)

        latent = visual_latent + context_projection
        return latent
