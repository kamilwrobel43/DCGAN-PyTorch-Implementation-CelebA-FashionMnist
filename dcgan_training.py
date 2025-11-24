import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from visualisations import plot_gen_imgs
from utils import generate_noise


def train_epoch(generator: nn.Module, discriminator: nn.Module, train_loader: dataloader, loss_fn, generator_optim: torch.optim, discriminator_optim: torch.optim, k_steps: int = 1, batch_size: int = 32, device: torch.device = torch.device('cuda'), label_smoothing_rate: float = 1.0):
    discriminator.train()
    generator.train()

    total_discriminator_loss, total_generator_loss = 0.0, 0.0

    for _, (img,label) in enumerate(train_loader):
        img = img.to(device)
        for _ in range(k_steps):
            z = generate_noise(batch_size, device)
            z = z.view(batch_size, 100, 1, 1)
            fake_img = generator(z).to(device)

            discriminator_optim.zero_grad()

            fake_y = discriminator(fake_img.detach())
            fake_loss = loss_fn(fake_y, torch.zeros_like(fake_y, device=device))


            real_y = discriminator(img)
            real_loss = loss_fn(real_y, torch.ones_like(real_y, device=device)*label_smoothing_rate)

            discriminator_loss = fake_loss + real_loss
            discriminator_loss.backward()
            discriminator_optim.step()

            total_discriminator_loss += discriminator_loss.item()


        generator_optim.zero_grad()
        z = generate_noise(batch_size, device)
        z = z.view(batch_size, 100, 1, 1)
        fake_img = generator(z).to(device)
        fake_y = discriminator(fake_img)
        generator_loss = loss_fn(fake_y, torch.ones_like(fake_y, device=device))
        generator_loss.backward()
        generator_optim.step()

        total_generator_loss += generator_loss.item()

    avg_discriminator_loss = total_discriminator_loss / (len(train_loader) * k_steps)
    avg_generator_loss = total_generator_loss / len(train_loader)

    return avg_generator_loss, avg_discriminator_loss

def train_dcgan(generator: nn.Module, discriminator: nn.Module, train_loader: dataloader, fixed_noise_list, loss_fn, generator_optim: torch.optim, discriminator_optim: torch.optim, k_steps: int = 1, batch_size: int = 32, n_epochs : int = 10, device = torch.device('cuda'), plot_freq: int = 10, label_smoothing_rate: float = 1.0):
    generator.to(device)
    discriminator.to(device)

    for epoch in range(1, n_epochs+1):
        generator_loss, discriminator_loss = train_epoch(generator, discriminator, train_loader,loss_fn, generator_optim, discriminator_optim, k_steps, batch_size, device, label_smoothing_rate)
        print(f"Epoch {epoch}/{n_epochs}: generator_loss: {generator_loss:.4f}| discriminator_loss: {discriminator_loss:.4f}")

        if epoch % plot_freq == 0:
            plot_gen_imgs(generator, device, fixed_noise_list)


    return generator, discriminator





