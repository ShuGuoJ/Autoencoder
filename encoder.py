import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
batchsz = 128

train_dataset = datasets.MNIST('../data', transform=transforms.ToTensor())
test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

encoder = torch.load('model/encoder_49.pkl')

def encoding(encoder, loader):
    """

    :param encoder: 编码器
    :param loader: 数据集
    :return: 中间representation
    """
    encoder = encoder.to(device)
    representation = []
    label = []
    for input, target in loader:
        input = input.to(device)
        input = input.view(-1, 784)
        with torch.no_grad():
            r = encoder(input)

        representation.append(r.cpu())
        label.append(target)

    # representation:[n_samples, n_dim]
    representation = torch.cat(representation, dim=0)
    # representation:[n_samples]
    label = torch.cat(label, dim=0)
    return representation, label

if __name__=='__main__':
    loaders = [train_loader, test_loader]
    save_path = ['train', 'test']
    for i, loader in enumerate(loaders):
        representation, label = encoding(encoder, loader)
        representation, label = representation.detach().numpy(), label.detach().numpy()
        np.savez('mnist_{}_representation'.format(save_path[i]), r=representation, l=label)

