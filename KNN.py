import torch
from torch import nn

class KNN(nn.Module):
    def __init__(self, data, label, k_neighbor=1):
        super(KNN, self).__init__()
        self.data = data.float()
        self.label = label
        self.k_neighbor = int(k_neighbor)
        self.max_index = int(torch.max(label).item())
        self.min_index = int(torch.min(label).item())

    def forward(self, x):
        # 数据转化
        # x:[batch, ndim]
        if x.ndim==1:
            x = x.unsqueeze(0)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        data = self.data.to(device)
        label = self.label.to(device)
        x = x.to(device)
        # data:[n_samples, ndim] => [n_samples, 1, ndim]
        # x:[batch, ndim] => [1, batch, ndim]
        data = data.unsqueeze(1)
        x = x.unsqueeze(0)
        # 计算距离
        d = data - x
        # d:[n_samples, batch]
        d = torch.norm(d, dim=-1)
        # 获取前k个邻居的label
        index = torch.argsort(d, dim=0)
        # top_k_index:[k_neighbor, batch]
        top_k_index = index[:self.k_neighbor, :]
        # top_k_index: [k_neighbor, batch] => [batch, neighbor]
        top_k_index = top_k_index.t().contiguous()
        neighborhood = torch.zeros(top_k_index.shape, dtype=torch.int)
        for i in range(top_k_index.shape[0]):
            neighborhood[i] = label[top_k_index[i]]
        # 统计label出现的次数
        length = self.max_index - self.min_index + 1
        count = torch.zeros((neighborhood.shape[0], length), device=device)
        for i in range(neighborhood.shape[0]):
            for j in neighborhood[i]:
                count[i][j-self.min_index] += 1
        # ans:[batch, 1]
        ans = torch.argmax(count, dim=-1, keepdim=True) + self.min_index
        return ans


# data = torch.tensor([[1,2],[2,2],[2,1],[-1,-2],[-2,-2],[-2,-1]],dtype=torch.float)
# label = torch.tensor([0,0,0,1,1,1], dtype=torch.int)
# knn = KNN(data, label, 2)
# x = torch.tensor([[0,1], [-1, 0]])
# print(knn(x))