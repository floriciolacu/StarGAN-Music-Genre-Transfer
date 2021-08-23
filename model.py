import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=64, num_speakers=4):
        super(Generator, self).__init__()
        conv_dim_copy = conv_dim
        layers = []
        layers.append(nn.ReflectionPad2d([3, 3, 3, 3]))
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(conv_dim_copy, conv_dim_copy * 2, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim_copy * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        conv_dim_copy = conv_dim_copy * 2
        layers.append(nn.Conv2d(conv_dim_copy, conv_dim_copy * 2, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim_copy * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        conv_dim_copy = conv_dim_copy * 2
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))
        layers.append(ResidualBlock(dim_in=conv_dim_copy, dim_out=conv_dim_copy))

        self.downsample = nn.Sequential(*layers)

        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 4 + num_speakers, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 2 + num_speakers, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.pad = nn.ReflectionPad2d([3, 3, 3, 3])
        self.upsample_3 = nn.Conv2d(conv_dim + num_speakers, 3, kernel_size=7, stride=1, bias=False, padding=0)

    def forward(self, x, c):
        x = self.downsample(x)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.upsample_1(x)
        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.upsample_2(x)
        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.pad(x)
        x = self.upsample_3(x)
        sigm = nn.Sigmoid()
        return sigm(x)


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, num_speakers=4):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3 + num_speakers, conv_dim, kernel_size=7, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(conv_dim, conv_dim * 4, kernel_size=7, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim * 4, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(conv_dim * 4, 1, kernel_size=7, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c.repeat(1, 1, x.size(2), x.size(3))], dim=1))


class Classifier(nn.Module):
    def __init__(self, conv_dim=64):
        super(Classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=[1, 12], stride=[1, 12]))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=[4, 1], stride=[4, 1]))
        layers.append(nn.InstanceNorm2d(conv_dim * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=[2, 1], stride=[2, 1]))
        layers.append(nn.InstanceNorm2d(conv_dim * 4, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=[8, 1], stride=[8, 1]))
        layers.append(nn.InstanceNorm2d(conv_dim * 8, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(conv_dim * 8, 4, kernel_size=[1, 7], stride=[1, 7]))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x.view(x.size(0), x.size(1))
