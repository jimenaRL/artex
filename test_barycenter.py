import numpy as np
import matplotlib.pylab as pl
import torch
import ot
import matplotlib.animation as animation

I1 = pl.imread("../../data/redcross.png").astype(np.float64)[::5, ::5, 2]
I2 = pl.imread("../../data/tooth.png").astype(np.float64)[::5, ::5, 2]

sz = I2.shape[0]
XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

x1 = np.stack((XX[I1 == 0], YY[I1 == 0]), 1) * 1.0
x2 = np.stack((XX[I2 == 0] + 60, -YY[I2 == 0] + 32), 1) * 1.0
x3 = np.stack((XX[I2 == 0], -YY[I2 == 0] + 32), 1) * 1.0

pl.figure(1, (8, 4))
pl.scatter(x1[:, 0], x1[:, 1], alpha=0.5)
pl.scatter(x2[:, 0], x2[:, 1], alpha=0.5)