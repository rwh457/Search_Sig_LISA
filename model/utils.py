import os
import fnmatch
import csv
import numpy as np


def print_dict(d: dict,  ncol: int, prex: str = ''):
    temp = [f"{k}: {v}" for k, v in d.items()]
    print(prex+('\n'+prex).join([', '.join(temp[i:: ncol]) for i in range(ncol)]))


def writer_row(p, filename, mode, contentrow):
    with open(p / filename, mode) as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(contentrow)


def ffname(path, pat):
    """Return a list containing the names of the files in the directory matches PATTERN.

    'path' can be specified as either str, bytes, or a path-like object.  If path is bytes,
        the filenames returned will also be bytes; in all other circumstances
        the filenames returned will be str.
    If path is None, uses the path='.'.
    'Patterns' are Unix shell style:
        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq
    Ref: https://stackoverflow.com/questions/33743816/how-to-find-a-
        filename-that-contains-a-given-string
    """
    return [filename
            for filename in os.listdir(path)
            if fnmatch.fnmatch(filename, pat)]


class Patching_data(object):
    """Patching for strain (the last dim)
    """

    def __init__(self, patch_size, overlap, sampling_frequency):
        """
        patch_size, sec
        overlap, sec
        """
        self.nperseg = int(patch_size * sampling_frequency)  # sec
        # noverlap must be less than nperseg.
        self.noverlap = int(overlap * self.nperseg)  # [%]
        # nstep = nperseg - noverlap
        print(f'\tPatching with patch size={patch_size}s and overlap={overlap}%.')

    def __call__(self, x):
        shape = x.shape
        # Created strided array of data segments
        if self.nperseg == 1 and self.noverlap == 0:
            return x[..., np.newaxis]
        # https://stackoverflow.com/a/5568169  also
        # https://iphysresearch.github.io/blog/post/signal_processing/spectral_analysis_scipy/#_fft_helper
        nstep = self.nperseg - self.noverlap
        shape = shape[:-1]+((shape[-1]-self.noverlap)//nstep, self.nperseg)
        strides = x.strides[:-1]+(nstep*x.strides[-1], x.strides[-1])
        return np.lib.stride_tricks.as_strided(x, shape=shape,
                                               strides=strides)