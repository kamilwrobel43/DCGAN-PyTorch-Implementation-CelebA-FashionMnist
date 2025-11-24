import torch
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
import numpy as np
from utils import generate_noise

def plot_img(x:torch.Tensor, discriminator_confidence, channels=1,):
    x = x.detach().cpu()
    x = (x + 1) / 2
    x = x.numpy()
    plt.figure(figsize=(4, 4))
    if channels == 1:
        # (H x W x 1) -> (H x W)
        plt.imshow(x.squeeze(), cmap='gray')
    else:
        plt.imshow(x)


    plt.axis('off')
    plt.title(f"Discriminator Prediction: {discriminator_confidence:.2f}")
    plt.show()

def plot_gen_imgs(generator, device, fixed_noise_list):
    plt.figure(figsize=(8, 3))
    generator = generator.to(device)
    generator.eval()
    with torch.no_grad():
        for i in range(10):
            plt.subplot(2,5,i+1)
            z = fixed_noise_list[i]
            z = z.view(1, 100, 1, 1)
            img_tensor = generator(z).squeeze(0)
            img_tensor = img_tensor.permute(1, 2, 0)

            np_img = img_tensor.detach().cpu().numpy()

            if np_img.ndim == 3 and np_img.shape[0] in [3, 4]:
                np_img = np.transpose(np_img, (1, 2, 0))
            elif np_img.ndim == 3 and np_img.shape[0] == 1:
                np_img = np_img.squeeze(0)

            if np_img.dtype != np.uint8:
                np_img = (np_img + 1) / 2

                np_img = np.clip(np_img, 0, 1)
                np_img = (np_img * 255).astype(np.uint8)

            cmap = 'gray' if np_img.ndim == 2 or (np_img.ndim == 3 and np_img.shape[-1] == 1) else None
            plt.imshow(np_img, cmap=cmap, interpolation='nearest')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_train_imgs(train_loader: dataloader):

    data_iter = iter(train_loader)
    images, labels = next(data_iter)


    plt.figure(figsize=(12, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        img_tensor = images[i]
        np_img = img_tensor.cpu().numpy()

        if np_img.ndim == 3 and np_img.shape[0] in [3, 4]:
            np_img = np.transpose(np_img, (1, 2, 0))
        elif np_img.ndim == 3 and np_img.shape[0] == 1:
            np_img = np_img.squeeze(0)

        if np_img.dtype != np.uint8:
            np_img = (np_img + 1) / 2

            np_img = np.clip(np_img, 0, 1)
            np_img = (np_img * 255).astype(np.uint8)


        cmap = 'gray' if np_img.ndim == 2 or (np_img.ndim == 3 and np_img.shape[-1] == 1) else None
        plt.imshow(np_img, cmap=cmap, interpolation='nearest')
        plt.axis('off')

    plt.tight_layout()
    plt.show()