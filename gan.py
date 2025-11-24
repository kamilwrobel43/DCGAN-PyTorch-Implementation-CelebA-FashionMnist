import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, channels: int = 1, img_height : int = 28, img_width : int = 28):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(100,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,256),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,channels*img_height*img_width),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.layers(z)
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1*28*28,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

