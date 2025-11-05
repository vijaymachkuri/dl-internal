# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:33:41 2025

@author: Admin
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 28*28
batch_size = 100
num_epochs = 50
learning_rate = 0.0002

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Scale images to [-1, 1]
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()  # Output scaled to [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output probability of real/fake
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=learning_rate)
optimizerG = optim.Adam(G.parameters(), lr=learning_rate)

# Labels
real_label = 1.
fake_label = 0.

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Flatten images to vectors
        images = images.view(-1, image_size).to(device)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        D.zero_grad()
        # Real images
        labels = torch.full((images.size(0),), real_label, dtype=torch.float, device=device)
        outputs = D(images).view(-1)
        lossD_real = criterion(outputs, labels)
        lossD_real.backward()

        # Fake images
        noise = torch.randn(images.size(0), latent_size, device=device)
        fake_images = G(noise)
        labels.fill_(fake_label)
        outputs = D(fake_images.detach()).view(-1)
        lossD_fake = criterion(outputs, labels)
        lossD_fake.backward()
        optimizerD.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        G.zero_grad()
        labels.fill_(real_label)  # We want generator to fool discriminator
        outputs = D(fake_images).view(-1)
        lossG = criterion(outputs, labels)
        lossG.backward()
        optimizerG.step()

        if i % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], '
                  f'Loss D: {lossD_real + lossD_fake:.4f}, Loss G: {lossG:.4f}')

    # Save generated images at the end of each epoch
    with torch.no_grad():
        fake_images = G(torch.randn(64, latent_size, device=device)).view(-1, 1, 28, 28)
        fake_images = (fake_images + 1) / 2  # Rescale to [0,1]
        grid = torchvision.utils.make_grid(fake_images, nrow=8)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f'Epoch {epoch+1}')
        plt.show()