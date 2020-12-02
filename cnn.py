import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


def accuracy(x, label):
    pred = torch.max(x, 1)[1]
    total_acc = pred.eq(label).sum() / x.shape[0]
    return total_acc


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=128, stride=1),
                                   nn.Sigmoid(), nn.AvgPool1d(kernel_size=4))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=1),
                                   nn.Sigmoid(), nn.AvgPool1d(kernel_size=3))
        self.out = nn.Sequential(nn.Linear(10400, 1500), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(1500, 200), nn.ReLU(),
                                 nn.Linear(200, 2))

    def forward(self, xi):
        xi = self.conv1(xi)
        xi = self.conv2(xi)
        xi = xi.view(xi.size(0), -1)
        out_put = self.out(xi)
        return out_put


if __name__ == '__main__':
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')
    # use whitened data in two channels
    train_data = train_data.reshape((6000, 2, 4096))

    x_data = torch.from_numpy(train_data)
    x_label = torch.from_numpy(train_label).long().reshape(-1)
    data_set = torch.utils.data.TensorDataset(x_data, x_label)
    train_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=128, shuffle=True)

    torch.manual_seed(1)
    LR = 0.001

    cnn = CNN().cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0)
    loss_func = nn.CrossEntropyLoss()
    EPOCH = 20

    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            batch_x, batch_y = data
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = cnn(batch_x)
            loss = loss_func(output, batch_y)
            print(loss)
            with torch.no_grad():
                print(accuracy(output, batch_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
