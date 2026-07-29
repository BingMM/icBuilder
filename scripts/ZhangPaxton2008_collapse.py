import numpy as np

from zhangpaxton2008 import zhang_paxton

mlt = np.arange(24.0)[None, :]
mlat = np.arange(50.0, 90.0)[:, None]
mean_energy, energy_flux = zhang_paxton(5.0, mlt, mlat)