# -*- coding: utf-8 -*-
"""
===============================================
Sliced Wasserstein Distance on 2D distributions
===============================================

.. note::
    Example added in release: 0.8.0.

This example illustrates the computation of the sliced Wasserstein Distance as
proposed in [31].

[31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of
measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pylab as pl
import numpy as np

import ot
import ipdb

##############################################################################
# Generate data
# -------------

# %% parameters and data generation

n = 200  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -0.8], [-0.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples


##############################################################################


def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(img):
    return np.clip(img, 0, 1)


# Loading images

I1 = pl.imread("ForLearning_2025/images/000078.png").astype(np.float64)[:1080, :1080, :]
I2 = pl.imread("ForLearning_2025/images/000132.png").astype(np.float64)[:1080, :1080, :]


x1 = im2mat(I1)
x2 = im2mat(I2)

pl.figure(1)
pl.plot(x1[:, 0], x1[:, 1], "+b", label="Source samples")
pl.plot(x2[:, 0], x2[:, 1], "xr", label="Target samples")
pl.legend(loc=0)
pl.title("Source and target distributions")

pl.show()

exit()

xs = x1
xt = x2

##############################################################################
# Plot data
# ---------

# %% plot samples

pl.figure(1)
pl.plot(xs[:, 0], xs[:, 1], "+b", label="Source samples")
pl.plot(xt[:, 0], xt[:, 1], "xr", label="Target samples")
pl.legend(loc=0)
pl.title("Source and target distributions")

n = 1080

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples



###############################################################################
# Sliced Wasserstein distance for different seeds and number of projections
# -------------------------------------------------------------------------

n_seed = 20
n_projections_arr = np.logspace(0, 3, 10, dtype=int)
res = np.empty((n_seed, 10))

# %% Compute statistics
for seed in range(n_seed):
    for i, n_projections in enumerate(n_projections_arr):
        res[seed, i] = ot.sliced_wasserstein_distance(
            xs, xt, a, b, n_projections, seed=seed
        )

res_mean = np.mean(res, axis=0)
res_std = np.std(res, axis=0)

###############################################################################
# Plot Sliced Wasserstein Distance
# --------------------------------

pl.figure(2)
pl.plot(n_projections_arr, res_mean, label="SWD")
pl.fill_between(
    n_projections_arr, res_mean - 2 * res_std, res_mean + 2 * res_std, alpha=0.5
)

pl.legend()
pl.xscale("log")

pl.xlabel("Number of projections")
pl.ylabel("Distance")
pl.title("Sliced Wasserstein Distance with 95% confidence interval")

pl.show()
