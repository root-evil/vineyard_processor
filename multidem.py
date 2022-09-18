from skimage import io
import matplotlib.pyplot as plt
import numpy as np


im = io.imread('data/KRA_PREC_100m.tif')

cnt = np.zeros((4096, 4096))
for i in range(0, 12):
    cnt += im[:, :, i]
cnt = cnt / 12

