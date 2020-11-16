import numpy as np
import pycbc.noise
import pycbc.types
from tqdm import tqdm


psd = np.load('psd_4096.npy')
Tobs = 61425.0
delta_t = 15.0
df = 1 / Tobs
tsamples = int(Tobs / delta_t) + 1
psd = pycbc.types.frequencyseries.FrequencySeries(psd, delta_f=df)


def noise_gen(N_s, dt, psd_v):
    noise = pycbc.noise.noise_from_psd(N_s, dt, psd_v)
    return noise


N = 4500
noise_matrix = np.zeros((N, tsamples))
for i in tqdm(range(N)):
    noise_matrix[i, :] = noise_gen(tsamples, delta_t, psd)

np.save('noise_matrix', noise_matrix)
