import numpy as np
import pycbc.noise
import pycbc.types
from tqdm import tqdm


def noise_gen(N_s, dt, psd_v):
    noise = pycbc.noise.noise_from_psd(N_s, dt, psd_v)
    return noise


if __name__ == '__main__':
    psd = np.load('psd_16384_15s.npy')
    Tobs = 16384 * 15.0
    delta_t = 15.0
    df = 1 / Tobs
    tsamples = int(Tobs / delta_t)
    psd = pycbc.types.frequencyseries.FrequencySeries(psd, delta_f=df)

    N = 12000
    noise_matrix = np.zeros((N, 2, tsamples))
    for i in tqdm(range(N)):
        noise_matrix[i, 0, :] = noise_gen(tsamples, delta_t, psd)
        noise_matrix[i, 1, :] = noise_gen(tsamples, delta_t, psd)
