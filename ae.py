import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, code_length=20):
        super(AE, self).__init__()

        # [b,784] => [b,20]
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,code_length),
            nn.ReLU(inplace=True)
        )

        # [b,20] => [b,784]
        self.decoder = nn.Sequential(
            nn.Linear(code_length,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    def forward(self,x):
        """
        :param x: [b,1,28,28]
        :return:
        """
        # flatten
        x = x.view(-1,784)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(-1,1,28,28)
        return x


# net = AE()
# input = torch.rand(1,1,28,28)
# out = net(input)
# print(out.shape)
# nn.BatchNorm1d