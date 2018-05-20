import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 256, 4, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 1, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        output = F.sigmoid(output + self.output_bias)
        return output


class Encoder(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.latent_size, 1, stride=1, bias=True)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output, x3.view(batch_size, -1), x2.view(batch_size, -1), x1.view(batch_size, -1)


class Discriminator(nn.Module):

    def __init__(self, latent_size, dropout, output_size=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size

        self.infer_x = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, self.output_size, 1, stride=1, bias=True)

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        if self.output_size == 1:
            output = F.sigmoid(output)
        return output.squeeze(), output_features.view(x.size()[0], -1)
