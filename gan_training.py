import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import matplotlib.pyplot as plt
from visualisations import plot_img
from utils import generate_noise


def train_epoch(generator: nn.Module, discriminator: nn.Module, train_loader: dataloader, loss_fn, generator_optim: torch.optim, discriminator_optim: torch.optim, k_steps: int = 1, batch_size: int = 32, device: torch.device = torch.device('cuda')):
    discriminator.train()
    generator.train()
    discriminator_loss, generator_loss = 0.0, 0.0

    for _, (img, label) in enumerate(train_loader):
        for _ in range(k_steps):
            z = generate_noise(batch_size, device)

            img = img.to(device)
            #(B x C x H x W) -> (B x C*H*W)
            img = img.view(img.size(0), -1)

            discriminator_optim.zero_grad()

            fake_img = generator(z).to(device)
            fake_y = discriminator(fake_img.detach())
            fake_loss = loss_fn(fake_y, torch.zeros_like(fake_y, device=device))

            real_y = discriminator(img)
            real_loss = loss_fn(real_y, torch.ones_like(real_y, device=device))

            discriminator_loss = fake_loss + real_loss
            discriminator_loss.backward()
            discriminator_optim.step()

        generator_optim.zero_grad()

        z = generate_noise(batch_size, device)
        fake_img = generator(z)
        fake_y = discriminator(fake_img)
        generator_loss = loss_fn(fake_y, torch.ones_like(fake_y, device=device))
        generator_loss.backward()
        generator_optim.step()

    return generator_loss, discriminator_loss


def train_gan(generator: nn.Module, discriminator: nn.Module, train_loader: dataloader, loss_fn, generator_optim: torch.optim, discriminator_optim: torch.optim, k_steps: int = 1, batch_size: int = 32, n_epochs : int = 10, device = torch.device('cuda'), height: int = 28, width: int = 28, channels: int = 1, plot_freq: int = 10):
    generator.to(device)
    discriminator.to(device)
    for epoch in range(1,n_epochs+1):
        generator_loss, discriminator_loss = train_epoch(generator, discriminator, train_loader,loss_fn, generator_optim, discriminator_optim, k_steps, batch_size, device)
        print(f"Epoch {epoch}/{n_epochs}: generator_loss: {generator_loss:.4f}| discriminator_loss: {discriminator_loss:.4f}")
        if epoch%plot_freq==0:
            z = generate_noise(batch_size=1, device=device).squeeze()
            generator.eval()
            with torch.no_grad():
                fake_img = generator(z)
                discriminator_confidence = discriminator(fake_img).item()
                fake_img = fake_img.view(channels, height, width)
                fake_img = fake_img.permute(1, 2, 0)
            plot_img(fake_img, discriminator_confidence)

    return generator, discriminator