import torch.nn as nn
class MNISTGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_proj1 = nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=7, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv_proj2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_proj3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        #(B,100,1,1) -> (B,128,7,7)
        x = self.conv_proj1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #(B,128,7,7) -> (B,64,14,14)
        x = self.conv_proj2(x)
        x = self.bn2(x)
        x = self.relu(x)

        #(B,64,14,14) -> (B,1,28,28)
        x = self.conv_proj3(x)
        x = self.tanh(x)

        return x

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        #(B,1,28,28) -> (B,64,14,14)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        #(B,64,14,14) -> (B,128,7,7)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        #(B,128,7,7) -> (B,1,1,1)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x



class CelebAGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_proj1 = nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU()

        self.conv_proj2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)


        self.conv_proj3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)


        self.conv_proj4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv_proj5 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # (B,100,1,1) -> (B,1024,4,4)
        x = self.conv_proj1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # (B,1024,4,4) -> (B,512,8,8)
        x = self.conv_proj2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # (B,512,8,8) -> (B,256,16,16)
        x = self.conv_proj3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # (B,256,16,16) -> (B,128,32,32)
        x = self.conv_proj4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # (B,128,32,32) -> (B,3,64,64)
        x = self.conv_proj5(x)
        x = self.tanh(x)

        return x


class CelebADiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # (B,3,64,64) -> (B,128,32,32)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        # (B,128,32,32) -> (B,256,16,16)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        # (B,256,16,16) -> (B,512,8,8)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        # (B,512,8,8) -> (B,1024,4,4)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        # (B,1024,4,4) -> (B,1,1,1)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x









