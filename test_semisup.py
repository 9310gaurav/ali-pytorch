import os
import argparse
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data
from sklearn.svm import LinearSVC
from model import *

batch_size = 64
latent_size = 256
cuda_device = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--samples_per_class', type=int, default=100)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def get_random_uniform_batch(data, targets, num_classes=10, samples_per_class=100):
    random_batch = np.zeros((num_classes*samples_per_class, data.shape[1]))
    random_targets = np.zeros(num_classes*samples_per_class)
    indices = np.random.permutation(data.shape[0])
    batch_size = 0
    label_counts = np.zeros(num_classes)
    for i in indices:
        if label_counts[targets[i]] < samples_per_class:
            label_counts[targets[i]] += 1
            random_batch[batch_size, :] = data[i, :]
            random_targets[batch_size] = targets[i]
            batch_size += 1
        if batch_size >= num_classes*samples_per_class:
            break

    return random_batch, random_targets


if __name__ == "__main__":

    encoder_state_dict = torch.load(opt.model_path)
    netE = Encoder(latent_size, True)
    netE.load_state_dict(encoder_state_dict)
    netE = tocuda(netE)

    print("Model restored")

    if opt.dataset == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=opt.dataroot, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
            batch_size=batch_size, shuffle=True)

    elif opt.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ])),
            batch_size=batch_size, shuffle=True)
    else:
        raise NotImplementedError

    all_embeddings = []
    all_targets = []

    for (data, target) in train_loader:
        temp, h1, h2, h3 = netE.forward(Variable(tocuda(data)))
        temp = np.concatenate([temp.view(data.size()[0], -1)[:latent_size, ].cpu().data.numpy(), h1.cpu().data.numpy(),
                               h2.cpu().data.numpy(), h3.cpu().data.numpy()], axis=1)
        all_embeddings.append(temp)
        all_targets.append(target.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    train_embeddings, validation_embeddings = all_embeddings[10000:, :], all_embeddings[:10000, :]
    train_targets, validation_targets = all_targets[10000:], all_targets[:10000]

    print("Embeddings calculated")

    random_batch, random_targets = get_random_uniform_batch(train_embeddings, train_targets)
    best_error_rate = 1.0
    best_C = None
    print(random_batch.shape)

    for log_C in np.linspace(-20, 20, 50):
        if log_C < -10 or log_C > 0:
            continue
        C = np.exp(log_C)
        svm = LinearSVC(C=C)
        svm.fit(random_batch, random_targets.ravel())
        error_rate = 1 - np.mean([
            svm.score(validation_embeddings[1000 * i:1000 * (i + 1), :],
                      validation_targets[1000 * i:1000 * (i + 1)].ravel())
            for i in range(10)
        ])
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_C = C
        print('C = {}, validation error rate = {} '.format(C, error_rate) +
              '(best is {}, {})'.format(best_C, best_error_rate))

    print("found best C : ", best_C)

    error_rates = []

    for j in range(1):
        random_batch, random_targets = get_random_uniform_batch(train_embeddings, train_targets)
        svm = LinearSVC(C=best_C)
        svm.fit(random_batch, random_targets.ravel())
        print(error_rates)
        error_rates.append(1 - np.mean([
            svm.score(validation_embeddings[1000 * i:1000 * (i + 1), :],
                      validation_targets[1000 * i:1000 * (i + 1)].ravel())
            for i in range(10)
        ]))

    print('Validation error rate = {} +- {} '.format(np.mean(error_rates),
                                                     np.std(error_rates)))