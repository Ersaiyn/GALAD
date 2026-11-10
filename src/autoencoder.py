"""Convolutional autoencoder architecture.

256x256x3 -> three stride-2 convs -> 32x32x128 -> FC bottleneck -> mirror back up.

The bottleneck is the one architectural choice that matters here. MSE is dominated
by bright pixels, so a tight bottleneck spends its capacity on luminosity rather
than on morphological structure -- which is the reason this detector works poorly
regardless of the setting. See REPORT.md section 4.2.
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, bottleneck=8):
        super().__init__()
        self.bottleneck = bottleneck
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc_encode = nn.Linear(128 * 32 * 32, bottleneck)
        self.fc_decode = nn.Linear(bottleneck, 128 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc_encode(x))
        x = torch.relu(self.fc_decode(x))
        x = x.view(x.size(0), 128, 32, 32)
        return self.decoder(x)
