import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.set_default_dtype(torch.float64)


def accuracy(x, label):
    pred = torch.max(x, 1)[1]
    total_acc = pred.eq(label).sum() / x.shape[0]
    return total_acc


class CNN(nn.Module):
    def __init__(self, S_t_m12, template, hh_sqrt):
        self.S_t_m12 = S_t_m12
        self.template = template
        self.hh_sqrt = hh_sqrt
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1),
                                   nn.Sigmoid(), nn.MaxPool1d(kernel_size=3))
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                                   nn.Sigmoid(), nn.MaxPool1d(kernel_size=3))
        self.out = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xi):
        # d / sqrt(S)
        xi = F.conv1d(xi, self.S_t_m12, padding=16383, groups=2)
        xi = torch.cat((xi, torch.zeros(xi.shape[0], 2, 16384 - xi.shape[-1] % 16384).cuda()), -1)
        d_SA = xi[:, 0, :].reshape(xi.shape[0], 2, 16384).sum(dim=1).reshape(-1, 1, 16384)
        d_SE = xi[:, 1, :].reshape(xi.shape[0], 2, 16384).sum(dim=1).reshape(-1, 1, 16384)

        h_SA = self.template[:, 0, :].reshape(-1, 1, 4096)
        h_SE = self.template[:, 1, :].reshape(-1, 1, 4096)

        # <d|h>
        dh_A = F.conv1d(d_SA, h_SA, padding=4095, groups=1)
        dh_E = F.conv1d(d_SE, h_SE, padding=4095, groups=1)
        dh_A = torch.cat((dh_A, torch.zeros(dh_A.shape[0], dh_A.shape[1], 16384 - dh_A.shape[-1] % 16384).cuda()), -1)
        dh_E = torch.cat((dh_E, torch.zeros(dh_E.shape[0], dh_E.shape[1], 16384 - dh_E.shape[-1] % 16384).cuda()), -1)
        dh_A = dh_A.reshape(dh_A.shape[0], dh_A.shape[1], 2, 16384).sum(dim=2)
        dh_E = dh_E.reshape(dh_E.shape[0], dh_E.shape[1], 2, 16384).sum(dim=2)
        xi = torch.stack((dh_A, dh_E), -1).permute(0, 2, 1, 3)
        xi = torch.div(xi, self.hh_sqrt)
        xi = torch.max(xi, dim=1).values.permute(0, 2, 1)

        xi = self.conv1(xi)
        xi = self.conv2(xi)
        xi = xi.view(xi.size(0), -1)
        out_put = self.out(xi)
        return out_put


if __name__ == '__main__':
    train_data = np.load('train_data_z3_random_AE_with_gb_16384.npy')
    train_label = np.zeros((6000,))
    train_label[3000:] = 1

    x_data = torch.from_numpy(train_data)
    x_label = torch.from_numpy(train_label).long()
    data_set = torch.utils.data.TensorDataset(x_data, x_label)
    train_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=128, shuffle=True)
    torch.manual_seed(1)
    LR = 0.001

    S_t_m12 = np.load('S_t.npy')[::-1].copy()
    S_t_m12 = torch.from_numpy(S_t_m12).reshape(1, 1, 16384)
    S_t_m12 = torch.cat((S_t_m12, S_t_m12), 0)
    template = np.load('template_St_matrix_z3_AE_4096_50.npy')
    template = torch.from_numpy(template).reshape(-1, 2, 4096)
    S_t_m12 = S_t_m12.cuda()
    template = template.cuda()
    hh_sqrt = np.load('hh_sqrt_4096_50.npy')
    hh_sqrt = torch.from_numpy(hh_sqrt).cuda()

    cnn = CNN(S_t_m12, template, hh_sqrt).cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0)
    loss_func = nn.CrossEntropyLoss()
    EPOCH = 20
    for epoch in tqdm(range(EPOCH)):
        for i, data in enumerate(train_loader):
            batch_x, batch_y = data
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            # batch_x = torch.autograd.Variable(batch_x)
            # batch_y = torch.autograd.Variable(batch_y)
            output = cnn(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)
        with torch.no_grad():
            print(accuracy(output, batch_y))
