import torch
import numpy as np
from KNN import KNN
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
datas = np.load('mnist_train_representation.npz')
train_data, train_label = torch.from_numpy(datas['r']), torch.from_numpy(datas['l'])
datas = np.load('mnist_test_representation.npz')
test_data, test_label = torch.from_numpy(datas['r']), torch.from_numpy(datas['l'])

class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

datasets = MyDataset(test_data, test_label)
batchsz = 128
loader = DataLoader(datasets, batch_size=batchsz, shuffle=True)
# KNN(data, label, k_neighbor)
knn = KNN(train_data, train_label, 10)

def compute_accuracy(net, loader):
    correct, total = 0, 0
    for input, label in loader:
        input, label = input.to(device), label.to(device)
        pred = net(input)
        if pred.ndim==2:
            pred = pred.squeeze(-1)
        correct += (pred==label).int().sum().item()
        total += input.shape[0]

    return correct/total

if __name__=='__main__':
    print('acc:{:.4f}'.format(compute_accuracy(knn, loader)))


