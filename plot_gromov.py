# -*- coding: utf-8 -*-
"""
==========================
Gromov-Wasserstein example
==========================

.. note::
    Example added in release: 0.8.0.

This example is designed to show how to use the Gromov-Wasserstein distance
computation in POT.
We first compare 3 solvers to estimate the distance based on
Conditional Gradient [24] or Sinkhorn projections [12, 51].
Then we compare 2 stochastic solvers to estimate the distance with a lower
numerical cost [33].

[12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
"Gromov-Wasserstein averaging of kernel and distance matrices".
International Conference on Machine Learning (ICML).

[24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
and Courty Nicolas
"Optimal Transport for structured data with application on graphs"
International Conference on Machine Learning (ICML). 2019.

[33] Kerdoncuff T., Emonet R., Marc S. "Sampled Gromov Wasserstein",
Machine Learning Journal (MJL), 2021.

[51] Xu, H., Luo, D., Zha, H., & Carin, L. (2019).
"Gromov-wasserstein learning for graph matching and node embedding".
In International Conference on Machine Learning (ICML), 2019.

"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#         Tanguy Kerdoncuff <tanguy.kerdoncuff@laposte.net>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ot

import numpy as np
import matplotlib.pylab as pl
import torch
import matplotlib.animation as animation
import cv2
import ipdb

import seaborn_image as isns
import matplotlib.colors as colors


#############################################################################
#
# Sample two Gaussian distributions (2D and 3D)
# ---------------------------------------------
#
# The Gromov-Wasserstein distance allows to compute distances with samples that
# do not belong to the same metric space. For demonstration purpose, we sample
# two Gaussian distributions in 2- and 3-dimensional spaces.


# n_samples = 30  # nb samples

# mu_s = np.array([0, 0])
# cov_s = np.array([[1, 0], [0, 1]])

# mu_t = np.array([4, 4, 4])
# cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# np.random.seed(0)
# xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
# np.random.seed(3536456)
# xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)

img = cv2.imread("ForLearning_2025/images/000078.png")
xs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.imread("ForLearning_2025/images/000132.png")
xt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #############################################################################
# #
# # Plotting the distributions
# # --------------------------

# ax1 = isns.imgplot(xs, describe=True, norm=colors.LogNorm())
# ax2 = isns.imgplot(xt, describe=True, norm=colors.LogNorm())

# pl.show()

# #############################################################################
# #
# # Compute distance kernels, normalize them and then display
# # ---------------------------------------------------------


# C1 = sp.spatial.distance.cdist(xs, xs)
# C2 = sp.spatial.distance.cdist(xt, xt)

# C1 /= C1.max()
# C2 /= C2.max()

# pl.figure(2)
# pl.subplot(121)
# pl.imshow(C1)
# pl.title("C1")

# pl.subplot(122)
# pl.imshow(C2)
# pl.title("C2")

# pl.show()


device = "cuda" if torch.cuda.is_available() else "cpu"

# I1 = pl.imread("redcross.png").astype(np.float64)[:, :, 2]
# I2 = pl.imread("tooth.png").astype(np.float64)[:, :, 2]
I1 = pl.imread("ForLearning_2025/images/000078.png").astype(np.float64)[:1080, :1080, 2]
I2 = pl.imread("ForLearning_2025/images/000132.png").astype(np.float64)[:1080, :1080, 2]

sz = I2.shape[0]
XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

th = 0.01
x1 = np.stack((XX[I1 > th], YY[I1 > th]), 1) * 1.0
x2 = np.stack((XX[I2 > th], YY[I2 > th]), 1) * 1.0
x3 = np.stack((XX[I2 > th], -YY[I2 > th] + 32), 1) * 1.0

pl.figure(1, (8, 4))
pl.scatter(x1[:, 0], x1[:, 1], alpha=0.5, color='r')
pl.scatter(x2[:, 0], x2[:, 1], alpha=0.5, color='g')
pl.show()


# use pyTorch for our data
x1_torch = torch.tensor(x1).to(device=device).requires_grad_(True)
x2_torch = torch.tensor(x2).to(device=device)

lr = 1e3
nb_iter_max = 500

x_all = np.zeros((nb_iter_max, x1.shape[0], 2))

loss_iter = []

# generator for random permutations
gen = torch.Generator(device=device)
gen.manual_seed(42)

for i in range(nb_iter_max):
    loss = ot.sliced_wasserstein_distance(
        x1_torch, x2_torch, n_projections=20, seed=gen
    )

    loss_iter.append(loss.clone().detach().cpu().numpy())
    loss.backward()

    # performs a step of projected gradient descent
    with torch.no_grad():
        grad = x1_torch.grad
        x1_torch -= grad * lr / (1 + i / 5e1)  # step
        x1_torch.grad.zero_()
        x_all[i, :, :] = x1_torch.clone().detach().cpu().numpy()

xb = x1_torch.clone().detach().cpu().numpy()

pl.figure(2, (8, 4))
pl.scatter(x1[:, 0], x1[:, 1], alpha=0.5, label="$\mu^{(0)}$")
pl.scatter(x2[:, 0], x2[:, 1], alpha=0.5, label=r"$\nu$")
pl.scatter(xb[:, 0], xb[:, 1], alpha=0.5, label="$\mu^{(100)}$")
pl.title("Sliced Wasserstein gradient flow")
pl.legend()
ax = pl.axis()
pl.show()