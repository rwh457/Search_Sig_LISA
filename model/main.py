# [W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.  # TODO
# http://www.diracprogram.org/doc/release-12/installation/mkl.html
# https://github.com/PaddlePaddle/Paddle/issues/17615
import os
import sys
import argparse
import csv
sys.path.insert(0, '..')
from model.dataset import (LISADatasetTorch, ToTensor)
from model.mfcnn import (MFLayer, CutHybridLayer)
from model.utils import (print_dict, writer_row, ffname)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt
# os.environ['OMP_NUM_THREADS'] = str(1)
# os.environ['MKL_NUM_THREADS'] = str(1)


class LISAModel(object):
    def __init__(self,
                 model_dir,
                 data_dir,
                 save_model_name,
                 use_cuda):
        super().__init__()
        if model_dir is None:
            raise NameError("Model directory must be specified."
                            " Store in attribute PosteriorModel.model_dir")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.save_model_name = save_model_name

        self.train_history = []
        self.test_history = []
        self.epoch_minimum_test_loss = 1
        self.epoch_cache = 1

        self.train_loader = None
        self.test_loader = None
        self.net = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_dataset(self, z, epoch_size, target_percentage, batch_size, num_workers, transform=None):
        ds_train = LISADatasetTorch(epoch_size, transform)
        ds_train.init_signals(z=z, target_percentage=target_percentage, istrain=True)
        self.train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))

        ds_test = LISADatasetTorch(epoch_size, transform)
        ds_test.init_signals(z=z, target_percentage=target_percentage, istrain=False)
        self.test_loader = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False, pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=lambda _: np.random.seed(
                int(torch.initial_seed()) % (2**32-1)))

    def init_model(self, template, hh_sqrt, S_t_m12):
        # the structure of the CNN
        self.net = nn.Sequential(
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
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LazyLinear(2),
            #nn.Sigmoid()
            # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        )

        self.net.to(self.device)

    def load_model(self):

        checkpoint = torch.load(self.model_dir / ffname(self.model_dir,
                                                        f'e*_{self.save_model_name}')[0],
                                map_location=self.device)
        # Load flow_net
        print('Load model from:', self.model_dir / ffname(self.model_dir,
                                                          f'e*_{self.save_model_name}')[0])

        self.net.load_state_dict(checkpoint['net_state_dict'])

        # Load loss history
        with open(self.model_dir / 'loss_history.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.train_history.append(float(row[1]))
                self.test_history.append(float(row[2]))

        # Set the epoch to the correct value. This is needed to resume
        # training.
        self.epoch_cache = checkpoint['epoch_cache']
        self.epoch_minimum_test_loss = checkpoint['epoch_minimum_test_loss']

    def init_training(self, kwargs):
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=kwargs['lr'])

        if kwargs['lr_annealing'] is True:
            if kwargs['lr_anneal_method'] == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=kwargs['steplr_step_size'],
                    gamma=kwargs['steplr_gamma'])
            elif kwargs['lr_anneal_method'] == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=kwargs['num_epochs'],
                )
            elif kwargs['lr_anneal_method'] == 'cosineWR':
                self.scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2
                    )
                )

    def train(self, num_epochs, output_freq):
        print('Starting timer')
        start_time = time.time()
        for epoch in range(self.epoch_cache, self.epoch_cache + num_epochs):
            # Update signals and noises in dataset
            self.train_loader.dataset.update()
            self.test_loader.dataset.update()

            print('Learning rate: {}'.format(
                self.optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss = self.train_epoch(epoch, output_freq)
            test_loss = self.test_epoch(epoch, output_freq)

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_history.append(train_loss)
            self.test_history.append(test_loss)

            self.epoch_cache = epoch + 1
            # Log/Plot/Save the history to file
            self._logging_to_file(epoch)
            if ((output_freq is not None) and (epoch == self.epoch_minimum_test_loss)):
                self._save_model(epoch)
        print('Stopping timer.')
        stop_time = time.time()
        print(f'Training time (including validation): {stop_time - start_time} seconds')

    def train_epoch(self, epoch, output_freq=50):
        """Train model for one epoch.
        """
        train_loss = 0.0
        self.net.train()

        start_time = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            if self.device is not None:
                y = y.to(torch.long).to(self.device, non_blocking=True)
                x = x.to(torch.float32).to(self.device, non_blocking=True)

            # Compute log prob
            loss = self.loss(self.net(x), y)

            # Keep track of total loss.
            train_loss += loss.sum()

            loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tCost: {:.2f}s'.format(
                    epoch, batch_idx *
                    self.train_loader.batch_size, len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(), time.time()-start_time))
                start_time = time.time()

        train_loss = train_loss.item() / len(self.train_loader.dataset)
        print(f'Train Epoch: {epoch} \tAverage Loss: {train_loss:.4f}')

        return train_loss

    def test_epoch(self, epoch, output_freq=50):
        """Calculate test loss for one epoch.
        """
        with torch.no_grad():
            self.net.eval()

            test_loss = 0.0
            start_time = time.time()
            for batch_idx, (x, y) in enumerate(self.test_loader):

                if self.device is not None:
                    y = y.to(torch.long).to(self.device, non_blocking=True)
                    x = x.to(torch.float32).to(self.device, non_blocking=True)

                # Compute log prob
                loss = self.loss(self.net(x), y)

                # Keep track of total loss
                test_loss += loss.sum()

                loss = loss.mean()
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tCost: {:.2f}s'.format(
                    epoch, batch_idx *
                    self.test_loader.batch_size, len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    loss.item(), time.time()-start_time))
                start_time = time.time()

            test_loss = test_loss.item() / len(self.test_loader.dataset)
            print(f'Test set: Average loss: {test_loss:.4f}')

            return test_loss

    @staticmethod
    def _plot_to(ylabel, p, filename):
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(p / filename)
        plt.close()

    def _logging_to_file(self, epoch):
        # Log the history to file

        # Make column headers if this is the first epoch
        if epoch == 1:
            writer_row(self.model_dir, 'loss_history.txt', 'w',
                       [epoch, self.train_history[-1], self.test_history[-1]])
        else:
            writer_row(self.model_dir, 'loss_history.txt', 'a',
                       [epoch, self.train_history[-1], self.test_history[-1]])

            data_history = np.loadtxt(self.model_dir / 'loss_history.txt')
            # Plot
            plt.figure()
            plt.plot(data_history[:, 0],
                     data_history[:, 1], '*--', label='train')
            plt.plot(data_history[:, 0],
                     data_history[:, 2], '*--', label='test')
            self._plot_to('Loss', self.model_dir, 'loss_history.png')
            self.epoch_minimum_test_loss = int(data_history[
                np.argmin(data_history[:, 2]), 0])

    def _save_model(self, epoch):
        for f in ffname(self.model_dir, f'e*_{self.save_model_name}'):
            os.remove(self.model_dir / f)
        print(f'Saving model as e{epoch}_{self.save_model_name}\n')
        self.save_model(filename=f'e{epoch}_{self.save_model_name}')

    def save_model(self, filename='model.pt'):
        cache_dict = {
            'script_args_dict': args,
            # 'hyperparams': self.net.model_hyperparams,  # TODO
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_cache': self.epoch_cache,
            'epoch_minimum_test_loss': self.epoch_minimum_test_loss,
        }

        if self.scheduler is not None:
            cache_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(cache_dict, self.model_dir / filename)


class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group, name = name.split('.', 1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Model the gravitational-wave parameter '
                     'posterior distribution with neural networks.'))

    # dir
    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--data_dir', type=str, default='../data')
    dir_parent_parser.add_argument('--save_model_name', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false', dest='cuda')

    # dataset: LISADatasetTorch
    dataset_parent_parser = argparse.ArgumentParser(add_help=None)
    dataset_parent_parser.add_argument(
        '--data.z', type=int, default=1)
    dataset_parent_parser.add_argument(
        '--data.target_percentage', nargs='+', type=float, default=(0.5, 0.8))
    dataset_parent_parser.add_argument(
        '--data.epoch_size', type=int, default='64')
    dataset_parent_parser.add_argument(
        '--data.batch_size', type=int, default='16')
    dataset_parent_parser.add_argument(
        '--data.num_workers', type=int, default='0')

    # train
    train_parent_parser = argparse.ArgumentParser(add_help=None)

    train_parent_parser.add_argument(
        '--train.num_epochs', type=int, default='10')
    train_parent_parser.add_argument(
        '--train.lr', type=float, default='0.001')
    train_parent_parser.add_argument(
        '--train.no_lr_annealing', action='store_false', dest='train.lr_annealing')
    train_parent_parser.add_argument(
        '--train.lr_anneal_method', choices=['step', 'cosine', 'cosineWR'], default='step')
    train_parent_parser.add_argument(
        '--train.steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument(
        '--train.steplr_step_size', type=int, default=80)
    train_parent_parser.add_argument(
        '--train.output_freq', type=int, default='20')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')
    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    # 1 ##      [train]/inference
    train_parser = mode_subparsers.add_parser(
        'train', description=('Train a network.'))

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    # 2.1 ##    [train] - [new]/existing
    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'),
        parents=[dir_parent_parser, dataset_parent_parser, train_parent_parser])

    # 2.2 ##    [train] - new/[existing]
    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, dataset_parent_parser, train_parent_parser])

    # 2.1.(1) (coupling function) [train] - [new]/existing
    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = False  # TODO

    ns = Nestedspace()
    return parser.parse_args(namespace=ns)


def main():
    global args
    args = parse_args()

    # print(args)

    print('Model directory\t', args.model_dir)
    print('Data directory\t', args.data_dir)
    lm = LISAModel(model_dir=args.model_dir,
                   data_dir=args.data_dir,
                   save_model_name=args.save_model_name,
                   use_cuda=args.cuda)
    print(f'Save the model as\t{args.save_model_name}')
    print('Device\t\t\t', lm.device)

    print('Init Dataset...')
    print_dict(vars(args.data), 5, '\t')
    transform = transforms.Compose([
        ToTensor(),
    ])
    lm.init_dataset(args.data.z,
                    args.data.target_percentage,
                    args.data.epoch_size,
                    args.data.batch_size,
                    args.data.num_workers,
                    transform)

    S_t_m12 = np.load(lm.data_dir / 'S_t.npy')[np.newaxis, np.newaxis, ...]
    S_t_m12 = torch.tensor(S_t_m12, dtype=torch.float32)  # [1, 1, 16384]

    template = np.load(lm.data_dir / 'template_St_matrix_z3_AE_4096_50.npy')
    template = torch.tensor(template, dtype=torch.float32)  # (50, 2, 4096) [4096 x 15]sec

    hh_sqrt = np.load(lm.data_dir / 'hh_sqrt_4096_50.npy')  # (50, 2)
    hh_sqrt = torch.tensor(hh_sqrt, dtype=torch.float32)

    if args.model_source == 'new':
        print('Init Model...')
        lm.init_model(template, hh_sqrt, S_t_m12)
    elif args.model_source == 'existing':
        lm.init_model(template, hh_sqrt, S_t_m12)
        lm.load_model()

    optimization_kwargs = dict(
        num_epochs=args.train.num_epochs,
        lr=args.train.lr,
        lr_annealing=args.train.lr_annealing,
        lr_anneal_method=args.train.lr_anneal_method,
        steplr_step_size=args.train.steplr_step_size,
        steplr_gamma=args.train.steplr_gamma,
    )
    lm.init_training(optimization_kwargs)
    print('\tArgumentations for training:')
    print_dict(vars(args.train), 3, '\t\t')

    try:
        lm.train(args.train.num_epochs, args.train.output_freq)
    except KeyboardInterrupt as e:
        print(e)
    finally:
        print('Finished!')


if __name__ == "__main__":
    main()
