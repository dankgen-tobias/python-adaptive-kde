from typing import Tuple

import numpy as np
from numba import jit

from sklearn.neighbors import KernelDensity


@jit(nopython=True)
def kernel_gauss_2d_normalize(x0, y0, x, y, err):
    """Returns the value of a normalized 2d gaussian kernel centered at (x0, y0) with a
    standard deviation of err at (x, y)"""
    return 1 / (2 * np.pi * err ** 2) * np.exp(-(
            ((x - x0) ** 2 / (2 * err ** 2))
            +
            ((y - y0) ** 2 / (2 * err ** 2))
    ))


@jit(nopython=True)
def kernel_gauss_2d(x0, y0, x, y, err, a=1):
    """Returns the value of a 2d gaussian kernel centered at (x0, y0) with a
    standard deviation of err and a height of a at (x, y)"""
    return a * np.exp(-(
            ((x - x0) ** 2 / (2 * err ** 2))
            +
            ((y - y0) ** 2 / (2 * err ** 2))
    ))


@jit(nopython=True)
def expensive_loop_adaptive(x_source, y_source, x_points, y_points, dir_err, z):
    """For each source source point add the value of normalized kernel to the grid """
    for i in range(len(x_source)):
        x_s = x_source[i]
        y_s = y_source[i]

        for xp in range(0, len(x_points)):
            for yp in range(0, len(y_points)):
                val = kernel_gauss_2d_normalize(x_points[xp], y_points[yp], x_s, y_s,
                                                dir_err[i])
                z[yp, xp] += val
    return z


@jit(nopython=True)
def expensive_loop(x_source, y_source, x_points, y_points, bandwidth, z):
    """For each source source point add the value of kernel to the grid."""

    for i in range(len(x_source)):
        x_s = x_source[i]
        y_s = y_source[i]

        for xp in range(0, len(x_points)):
            for yp in range(0, len(y_points)):
                # non-normalized kernel can be used since smoothing is constant
                val = kernel_gauss_2d(x_points[xp], y_points[yp], x_s, y_s, bandwidth)
                z[yp, xp] += val
    return z


@jit(nopython=True)
def expensive_loop_hist_smoothing(x_grid, y_grid, x_hist, y_hist,
                                  smoothing, z, hist_values):
    for ix_grid in range(len(x_grid)):
        for iy_grid in range(len(y_grid)):
            for ix_hist in range(len(x_hist)):
                for iy_hist in range(len(y_hist)):
                    val = kernel_gauss_2d(
                        x_hist[ix_hist], y_hist[iy_hist],
                        x_grid[ix_grid], y_grid[iy_grid],
                        smoothing, a=hist_values[iy_hist, ix_hist]
                    )
                    z[ix_grid, iy_grid] += val
    return z.T


def normalize_2d_data(data, sum):
    return data / np.sum(data) * sum


def smooth_histogram(
        hist: Tuple[np.array, np.array, np.array],
        x_grid: np.array,
        y_grid: np.array,
        smoothing: float
) -> np.array:
    x_hist = np.array(
        [(hist[1][i] + hist[1][i + 1]) / 2 for i in range(len(hist[1]) - 1)])
    y_hist = np.array(
        [(hist[2][i] + hist[2][i + 1]) / 2 for i in range(len(hist[2]) - 1)])
    x_grid = x_grid[0, :]
    y_grid = y_grid[:, 0]
    z = np.zeros((len(x_grid), len(y_grid)))

    z_new = expensive_loop_hist_smoothing(
        x_grid, y_grid, x_hist, y_hist, smoothing, z, hist[0].T
    )

    return z_new

class ownKDE:
    def __init__(self, grid=None):
        self.grid = grid

    def score_samples(
            self,
            x_source, y_source, dir_err,
            adaptive_bandwidth=True, **kwargs
    ):
        z = np.zeros(shape=np.shape(self.grid['x']))

        # get the x and y values of the map
        x_points = self.grid['x'][0, :]
        y_points = self.grid['y'][:, 0]

        # median of direction error
        if not adaptive_bandwidth:
            bandwidth = np.median(dir_err)
            z = expensive_loop(x_source, y_source, x_points, y_points, bandwidth, z)

        else:
            z = expensive_loop_adaptive(x_source, y_source, x_points, y_points, dir_err,
                                        z)

        return z


def kde_2d_scikit(x, y, xx_map, yy_map, bandwidth, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    # create grid of sample locations (default: 100x100)
    xy_sample = np.vstack([yy_map.ravel(), xx_map.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
    kde.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde.score_samples(xy_sample))
    return np.reshape(z, xx_map.shape)
