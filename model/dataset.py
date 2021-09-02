from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pycbc.noise
import pycbc.types
import pycbc.psd
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)


class LISADatasetTorch(torch.utils.data.Dataset):
    """LISA dataset

    Usage:
    >>> dataset = LISADatasetTorch(epoch_size = 32)
    >>> dataset.init_signals(z=3, istrain=True)
    >>> dataset.update()
    """

    def __init__(self, epoch_size,  # number of the train samples
                 transform=None,  # transform for data_block
                 data_dir='../data',
                 num_input=16384,  # num of input
                 delta_t=15):
        """
        Args:

        """
        Record = namedtuple('Record', 'epoch_size data_dir delta_t delta_f \
                             num_input, num_total num_signals \
                             n_dgb_A n_dgb_E n_igb_A n_igb_E')
        self.var = Record(epoch_size, Path(data_dir), delta_t, 1.0 / (num_input * delta_t),
                          num_input,  # num of samplss for input
                          2102401,  # num of samples in one year
                          epoch_size // 2,  # by default, 1/2 of epoch_size is signals
                          #  for noises without signals, half of them contain confusion noise
                          np.load(Path(data_dir) / 'confusion_noise' / 'dgb_tdi_A_15s.npy'),
                          np.load(Path(data_dir) / 'confusion_noise' / 'dgb_tdi_E_15s.npy'),
                          np.load(Path(data_dir) / 'confusion_noise' / 'igb_tdi_A_15s.npy'),
                          np.load(Path(data_dir) / 'confusion_noise' / 'igb_tdi_E_15s.npy'),)

        self.psd = None
        self.noise_block = None
        self.signal_block = None
        self.set_psd()
        self.transform = transform

    def __len__(self):
        return self.var.epoch_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform is not None:
            self.data_block[idx] = self.transform(self.data_block[idx])

        return (self.data_block[idx], self.label_block[idx])

    @property
    def data_block(self):
        return self.noise_block + np.pad(self.signal_block,
                                         ((0, self.var.epoch_size - self.var.num_signals),
                                          (0, 0),
                                          (0, 0)),
                                         mode='constant',
                                         constant_values=0)

    @property
    def label_block(self):
        return np.pad([1, 0],
                      (self.var.num_signals - 1,
                       self.var.epoch_size - self.var.num_signals - 1),
                      mode='edge')

    def init_signals(self, z, istrain=True):
        """Init and update signal_block
        """
        # TODO: num of samples in signal_block vs epoch_size
        cache = '' if istrain else 'test_'
        self.signal_block = np.load(self.var.data_dir / 'signal' / f'z{z}' /
                                    f'{cache}sig_z{z}_random_AE_16384.npy')  # (3000, 2, 16384)
        self.update_signals()

    def update_signals(self):
        """Update signals (reshuffle)
        """
        np.random.shuffle(self.signal_block)
        self.signal_block = self.signal_block[:self.var.num_signals]

    def update(self):
        """Update signals and noises (Gaussian + confusion)
        """
        self.update_noise()
        self.update_signals()

    def _init_noise(self):
        """Init a cache noise_block
        """
        self.noise_block = np.empty((self.var.epoch_size, 2, self.var.num_input), dtype=np.float64)

    def update_noise(self):
        """Regenerate Gaussian and confusion noise.
        """
        self._init_noise()
        self._add_gaussian_noise_block()
        self._add_conf_noise_block()

    def _add_conf_noise_block(self):
        index = (np.random.rand(self.var.epoch_size) *
                 (self.var.num_total - self.var.num_input)).astype(int)
        for i in tqdm(range(self.var.epoch_size), desc='Adding confusion noises', disable=False):
            self.noise_block[i, 0] += self.var.n_dgb_A[index[i]:(index[i] + self.var.num_input)]
            self.noise_block[i, 0] += self.var.n_igb_A[index[i]:(index[i] + self.var.num_input)]
            self.noise_block[i, 1] += self.var.n_dgb_E[index[i]:(index[i] + self.var.num_input)]
            self.noise_block[i, 1] += self.var.n_igb_E[index[i]:(index[i] + self.var.num_input)]

    def _add_gaussian_noise_block(self):
        for i in tqdm(range(self.var.epoch_size), desc='Adding Gaussian noises', disable=False):
            self.noise_block[i, 0] += self._gen_gaussian_noise()
            self.noise_block[i, 1] += self._gen_gaussian_noise()

    def _gen_gaussian_noise(self):
        """ Noise Generation (pycbc)
        """
        return pycbc.noise.noise_from_psd(self.var.num_input, self.var.delta_t, self.psd, seed=127)

    def set_psd(self, filename='psd.txt', length=8193, low_freq_cutoff=1e-5):
        """ PSD
        """
        self.psd = pycbc.psd.from_txt(self.var.data_dir / filename, length=length,
                                      delta_f=self.var.delta_f, low_freq_cutoff=low_freq_cutoff,
                                      is_asd_file=False)


if __name__ == "__main__":
    wfd = LISADatasetTorch(20)
    wfd.init_signals(z=3)

    from torch.utils.data import DataLoader

    batch_size = 4
    num_workers = 1
    train_loader = DataLoader(
        wfd, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2**32-1)))
    train_loader.dataset.update()

    for x, y in train_loader:
        break
    print(x.shape, y.shape)

    x, y = wfd[:10]
    print(x.shape, y.shape)
