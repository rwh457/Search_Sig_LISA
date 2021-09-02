import torch
from torch import nn
import torch.nn.functional as F


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
        return F.pad(X, pad=(0, (-X.shape[-1]) % mod)).\
                 unsqueeze(-2).\
                 reshape(X.shape[0], -1, abs((-X.shape[-1]) // mod), mod).\
                 sum(-2)

    def forward(self, X):
        # split A & E
        xa = X[:, :1]
        xe = X[:, 1:]

        # d / sqrt(S)
        d_SA = self._mod(F.conv1d(xa, self.params['S_t_m12'], padding=self.data_size - 1, groups=1),
                         mod=self.data_size)
        d_SE = self._mod(F.conv1d(xe, self.params['S_t_m12'], padding=self.data_size - 1, groups=1),
                         mod=self.data_size)
        # [num_batch, 1, self.data_size]

        h_SA = self.params['template'][:, :1]
        h_SE = self.params['template'][:, 1:]
        # [num_temp, 1, self.temp_size]

        # <d|h>
        dh_A = self._mod(F.conv1d(d_SA, h_SA, padding=self.temp_size - 1, groups=1),
                         mod=self.data_size)
        dh_E = self._mod(F.conv1d(d_SE, h_SE, padding=self.temp_size - 1, groups=1),
                         mod=self.data_size)

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
