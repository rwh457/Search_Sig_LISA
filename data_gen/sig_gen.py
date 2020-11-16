import numpy as np
from GenerateFD_SignalTDIs import ComputeMBHBXYZ_FD
from tqdm import tqdm

z = 3
M = np.power(10, 5.4 + np.arange(0, 200) * 0.008)
Mz = M * (1 + z)
q = np.arange(1, 16)
tc = 4096 * 15.0 * (39.6 + np.random.rand(3000) * 0.3)
chi1s = 0.0
chi2s = 0.0
Stheta1s = 0.0
Stheta2s = 0.0
phi0 = 6.24789726
bet = 0.292696
lam = 3.5091
inc = 1.2245321
psi = -0.2044459
dist = 56005.7836628
Tobs = 2457600
del_t = 15.0

k = 0
sig_matrix = np.zeros((3000, 4096))

for i in tqdm(range(Mz.size)):
    for j in range(q.size):
        m1 = q[j] * Mz[i] / (1 + q[j])
        m2 = Mz[i] / (1 + q[j])
        m1 = np.around(m1, decimals=4)
        m2 = np.around(m2, decimals=4)
        freq, Xf, Yf, Zf = ComputeMBHBXYZ_FD(m1, m2, chi1s, chi2s, Stheta1s, Stheta2s, tc[k], phi0, bet, lam, inc,
                                             psi, dist, Tobs, del_t, verbose=False)

        Xta = np.fft.irfft(Xf) * (1.0 / del_t)
        Yta = np.fft.irfft(Yf) * (1.0 / del_t)
        Zta = np.fft.irfft(Zf) * (1.0 / del_t)

        Xt = Xta[159744:163840]
        Yt = Yta[159744:163840]
        Zt = Zta[159744:163840]
        At = (2 * Xt - Yt - Zt) / 3
        sig_matrix[k, :] = At

        k = k + 1

np.save('sig_matrix', sig_matrix)
