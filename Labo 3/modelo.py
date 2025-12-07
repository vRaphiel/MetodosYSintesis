import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        b = x.shape[0]
        x = x.view(b, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = decoded.view(b, 1, 28, 28)
        return decoded