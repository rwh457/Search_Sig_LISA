import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pycbc.noise
import pycbc.types


def accuracy(x, label):
    pred = torch.max(x, 1)[1]
    total_acc = pred.eq(label).sum() / x.shape[0]
    return total_acc


def noise_gen(N_s, dt, psd_v):
    noise = pycbc.noise.noise_from_psd(N_s, dt, psd_v)
    return noise


S_t_m12 = np.load('S_t.npy')[::-1].copy()
S_t_m12 = torch.tensor(S_t_m12, dtype=torch.float32).reshape(1, 1, 16384)

template = np.load('template_St_matrix_z3_AE_4096_50.npy')
template = torch.tensor(template, dtype=torch.float32).reshape(-1, 2, 4096)  # (50, 2, 4096) [4096 x 15]sec

hh_sqrt = np.load('hh_sqrt_4096_50.npy')  # (50, 2)
hh_sqrt = torch.tensor(hh_sqrt, dtype=torch.float32)


class MFLayer(nn.Module):
    def __init__(self, template, hh_sqrt, S_t_m12):
        super(MFLayer, self).__init__()
        self.data_size = S_t_m12.shape[-1]  # 16384
        self.temp_size = template.shape[-1]  # 4096
        self.params = nn.ParameterDict({
            'template': nn.Parameter(template, requires_grad=False),
            'hh_sqrt': nn.Parameter(hh_sqrt.unsqueeze(0).unsqueeze(-1), requires_grad=False),
            'S_t_m12': nn.Parameter(S_t_m12, requires_grad=False),
        })

    def _mod(self, X, mod):
        return F.pad(X, pad=(0, (-X.shape[-1]) % mod)).unsqueeze(-2).reshape(X.shape[0], -1, abs((-X.shape[-1]) // mod),
                                                                             mod).sum(-2)

    def forward(self, X):
        # split A & E
        xa = X[:, :1]
        xe = X[:, 1:]

        # d / sqrt(S)
        d_SA = self._mod(F.conv1d(xa, self.params['S_t_m12'], padding=self.data_size - 1, groups=1), mod=self.data_size)
        d_SE = self._mod(F.conv1d(xe, self.params['S_t_m12'], padding=self.data_size - 1, groups=1), mod=self.data_size)
        # [num_batch, 1, self.data_size]

        h_SA = self.params['template'][:, :1]
        h_SE = self.params['template'][:, 1:]
        # [num_temp, 1, self.temp_size]

        # <d|h>
        dh_A = self._mod(F.conv1d(d_SA, h_SA, padding=self.temp_size - 1, groups=1), mod=self.data_size)
        dh_E = self._mod(F.conv1d(d_SE, h_SE, padding=self.temp_size - 1, groups=1), mod=self.data_size)

        # [num_batch, num_temp, 2, self.data_size]
        return torch.stack((dh_A, dh_E), -2) / self.params['hh_sqrt']


class CutHybridLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.max(torch.abs(X), -1).values.permute(0, 2, 1)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.flatten(X, start_dim=1)


class NormalizeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        #         mean = torch.mean(X, dim=-1, keepdim=True)
        #         std = torch.std(X, dim=-1, keepdim=True)
        #         X = (X - mean) / std
        return X.sqrt()


if __name__ == '__main__':
    # the structure of the CNN
    net = nn.Sequential(
        MFLayer(template, hh_sqrt, S_t_m12),
        CutHybridLayer(),
        #     NormalizeLayer(),
        nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=4, stride=2),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=4, stride=2),
        nn.Flatten(),
        #     Flatten(),
        nn.Linear(288, 32),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(32, 2),
        nn.Sigmoid()
    )

    # noise generation
    psd = np.load('psd_16384_15s.npy')
    Tobs = 16384 * 15.0
    delta_t = 15.0
    df = 1 / Tobs
    tsamples = int(Tobs / delta_t)
    psd = pycbc.types.frequencyseries.FrequencySeries(psd, delta_f=df)

    N_train = 6000  # number of the train samples
    train_data = np.zeros((N_train, 2, tsamples))  # (6000, 2, 16384) [16384 * 15]sec
    for i in range(N_train):
        train_data[i, 0, :] = noise_gen(tsamples, delta_t, psd)
        train_data[i, 1, :] = noise_gen(tsamples, delta_t, psd)

    #  for noises without signals (3000 samples), half of them contain confusion noise
    n_dgb_A = np.load('./confusion_noise/dgb_tdi_A_15s.npy')  # [2102401 * 15]sec
    n_dgb_E = np.load('./confusion_noise/dgb_tdi_E_15s.npy')  # [2102401 * 15]sec
    n_igb_A = np.load('./confusion_noise/igb_tdi_A_15s.npy')  # [2102401 * 15]sec
    n_igb_E = np.load('./confusion_noise/igb_tdi_E_15s.npy')  # [2102401 * 15]sec
    index = (np.random.rand(1500) * 2086017).astype(int)
    for i in range(1500):
        train_data[i, 0, :] = train_data[i, 0, :] + n_dgb_A[index[i]:(index[i] + 16384)] \
                              + n_igb_A[index[i]:(index[i] + 16384)]
        train_data[i, 1, :] = train_data[i, 1, :] + n_dgb_E[index[i]:(index[i] + 16384)] \
                              + n_igb_E[index[i]:(index[i] + 16384)]

    #  signal injection (3000, with confusion noise)
    z = 3
    sig = np.load('./signal/z{}/sig_conf_z{}_random_AE_16384.npy'.format(z, z))
    train_data[3000:, :, :] = train_data[3000:, :, :] + sig

    train_label = np.zeros((6000,))
    train_label[3000:] = 1

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.long)

    batch_size = 128
    dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                              worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)))

    # test data used in training
    test_data = np.load('test_LDC_16384_plus.npy')  # (450, 2, 16384)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    #  half of the test data contain GW signals
    test_label = np.zeros((450,))
    test_label[92:94] = 1
    test_label[90:92] = 1
    test_label[133:135] = 1
    test_label[239:241] = 1
    test_label[150:152] = 1
    test_label[89:81] = 1
    test_label[70:72] = 1
    test_label[139:141] = 1
    test_label[197:199] = 1
    test_label[165:167] = 1
    test_label[189:191] = 1
    test_label[96:98] = 1
    test_label[179:181] = 1
    test_label[38:40] = 1
    test_label[109:110] = 1
    test_label[255:] = 1
    test_label = torch.tensor(test_label, dtype=torch.long)

    test_data = test_data.cuda()
    test_label = test_label.cuda()

    # training
    net = net.cuda()

    lr, wd = 1e-3, 1e-4
    lr_period, lr_decay = 2, 0.9
    num_epochs = 25

    trainer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad),
                               lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches = len(train_loader)
    loss = nn.CrossEntropyLoss(reduction="none")

    accuracy_list = np.zeros(num_epochs)
    loss_list = np.zeros(num_epochs)

    for epoch in tqdm(range(num_epochs)):
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.cuda(), labels.cuda()
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).mean()
            l.backward()
            trainer.step()

        loss_list[epoch] = l.item()

        with torch.no_grad():
            tmp = accuracy(net(test_data), test_label).item()
            accuracy_list[epoch] = tmp

        # if the accuracy is larger than the previous values, the model parameters will be saved
        if epoch > 0 and tmp > np.max(accuracy_list[0:epoch]):
            check_point = {'epoch': epoch, 'model_state_dict': net.state_dict()}
            path_checkpoint = './model_param/checkpoint_{}_epoch.pkl'.format(epoch)
            torch.save(check_point, path_checkpoint)

        tqdm.write('loss is {} and accuracy is {}'.format(l.item(), tmp))
        scheduler.step()

    np.save('loss', loss_list)
    np.save('accuracy', accuracy_list)
