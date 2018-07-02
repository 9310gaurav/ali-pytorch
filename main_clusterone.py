import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model import *
import os
from pathlib import Path
from clusterone import get_data_path, get_logs_path


CLUSTERONE_USERNAME = "gaurav9310"

batch_size = 100
lr = 1e-4
latent_size = 256
num_epochs = 100
cuda_device = "0"


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', required=True)
parser.add_argument('--save_image_dir', required=True)
parser.add_argument('--reuse', type=boolean_string, default=False)
parser.add_argument('--save_freq', type=int, default=1)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))




train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=get_data_path(
        dataset_name="%s/cifars3"%CLUSTERONE_USERNAME,
        local_root=opt.dataroot,
        local_repo="",
        path=""
    )
                     , train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor()
                  ])),
    batch_size=batch_size, shuffle=True)

save_image_dir = get_logs_path(opt.save_image_dir)
save_model_dir = get_logs_path(opt.save_model_dir)

netE = tocuda(Encoder(latent_size, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

if opt.reuse:
    for epoch in range(num_epochs):
        if epoch % opt.save_freq == 0:
            model_file = Path("%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
            if model_file.is_file():
                continue
            else:
                break

    epoch = epoch - 1*opt.save_freq

    if epoch == -1*opt.save_freq:
        netE.apply(weights_init)
        netG.apply(weights_init)
        netD.apply(weights_init)
        print("No saved models found to resume from. Starting from scratch.")
    else:
        print("Loading models saved after epochs : ", epoch + 1)
        encoder_state_dict = torch.load("%s/netE_epoch_%d.pth" % (save_model_dir, epoch))
        generator_state_dict = torch.load("%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
        discriminator_state_dict = torch.load("%s/netD_epoch_%d.pth" % (save_model_dir, epoch))

        netE.load_state_dict(encoder_state_dict)
        netG.load_state_dict(generator_state_dict)
        netD.load_state_dict(discriminator_state_dict)
else:
    netE.apply(weights_init)
    netG.apply(weights_init)
    netD.apply(weights_init)

current_epoch = epoch + 1*opt.save_freq
optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(current_epoch, num_epochs):

    i = 0
    for (data, target) in train_loader:

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label)

        if loss_g.data[0] < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        if i % 100 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.data[0], "G loss :", loss_g.data[0],
                  "D(x) :", output_real.mean().data[0], "D(G(x)) :", output_fake.mean().data[0])

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ],
                              "%s/fake.png" % (save_image_dir)
                              )
            vutils.save_image(d_real.cpu().data[:16, ], "%s/real.png"% (save_image_dir))

        i += 1

    if epoch % opt.save_freq == 0:
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (save_model_dir, epoch))
        torch.save(netE.state_dict(), "%s/netE_epoch_%d.pth" % (save_model_dir, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (save_model_dir, epoch))

        vutils.save_image(d_fake.cpu().data[:16, ], "%s/fake_%d.png" % (save_image_dir, epoch))
