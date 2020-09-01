import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ae import AE
from visdom import Visdom

device = torch.device('cuda:0')  if torch.cuda.is_available() else torch.device('cpu')
batchsz = 128
epochs = 50
lr = 1e-3

train_dataset = datasets.MNIST('../data',transform=transforms.ToTensor())
test_dataset = datasets.MNIST('../data', False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)

net = AE()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 10)
net.to(device)
criterion.to(device)
train_loss = []
viz = Visdom()
for epoch in range(epochs):
    train_loss.clear()
    net.train()
    for step, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x_hat = net(x)

        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if step%50==0:
            print('epoch:{} batch:{} loss:{:.6f}'.format(epoch, step, loss.item()))

    scheduler.step()
    net.eval()
    x, _= next(iter(test_loader))
    x = x.to(device)
    x_hat = net(x)
    viz.images(x, nrow=6, win='x', opts=dict(title='x'))
    viz.images(x_hat, nrow=6, win='x_hat', opts=dict(title='x_hat'))
    if (epoch+1)%10==0:
        torch.save(net.encoder, 'model/encoder_{}.pkl'.format(epoch))