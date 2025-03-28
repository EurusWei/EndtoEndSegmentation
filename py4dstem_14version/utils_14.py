#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:20:16 2022

@author: weiliu
"""

import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM
from os import path
import math
from scipy import optimize
from scipy import linalg
import matplotlib.pyplot as plt
import warnings
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like, ListedColormap
from numpy.ma import MaskedArray
from numbers import Number
from math import log
from copy import copy
import time
from sklearn.decomposition import PCA

# ref: https://levelup.gitconnected.com/a-simple-method-to-calculate-circular-intensity-averages-in-images-4186a685af3


def radial_curve(dp, center):
    """
    Parameters
    ----------
    dp : DP data in np.array.
    center : [i_0, j_0], this center is consistent for the same nanoparticle data.

    Returns
    -------
    radius data, and corresponding intensitiies in np.array format.
    """
    ## center: [center[0], center[1]] = [73,56]
    row, col = dp.shape
    X, Y = np.meshgrid(np.arange(row), np.arange(col))
    R = np.sqrt(np.square(X - center[1]) + np.square(Y - center[0]))
    rad = np.arange(1, np.max(R), 0.25)

    intensity = np.zeros(len(rad))
    index = 0

    bin_size = 1
    for i in rad:
        im_mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
        values = dp[im_mask]
#         if i == 1:
#             plt.imshow(im_mask)
        intensity[index] = np.mean(values)
        index += 1
#     print(intensity[0])
    return rad, intensity


def show_radial(datacube, Rx, Ry, center):
    # dp = datacube_symmetrize.data[]
    rad, intensity = radial_curve(datacube.data[Rx][Ry], center=center)
    # Create figure and add subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot data
    ax.plot(rad, intensity, linewidth=2)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 20)


def projection(braggpeaks_intensity, K_max, datacube_symmetrize, particle_labels):
    '''
    Note that during projection, excluding the real space info (first two columns).
    '''
    x = braggpeaks_intensity[particle_labels == 1]
    pca = PCA(n_components=K_max)
    X_project = pca.fit_transform(x)
    # transform the 2D reciprocal space features to 3D reciprocal space features
    transformed_features = np.zeros((datacube_symmetrize.data.shape[0], datacube_symmetrize.data.shape[1],
                                     X_project.shape[-1]))
    transformed_features[particle_labels == 1] = X_project
    beta = find_beta(datacube_symmetrize,
                     transformed_features, particle_labels)
    # obtain 2D coordinate matrix of scan positions and concatenate to reciprocal space feature matrix
    real_pos = np.array(np.where(particle_labels == 1)).T
    X_project = np.hstack((real_pos, beta * X_project))
    return X_project, beta, pca


def average_datacube(datacube, n):
    """
        Smoothing operation to improve the quality of DPs;
        Note that datacube makes changes in-place, so store the data in separate
    array for the usage of later checks.
    """
    datacube_average = datacube
    avg = np.zeros((datacube.R_Nx - 2 * n, datacube.R_Ny -
                    2 * n, datacube.Q_Ny, datacube.Q_Ny))
    for i in range(n, datacube.R_Nx - n):
        for j in range(n, datacube.R_Ny - n):
            avg[i - n, j - n, :,
                :] = np.mean(datacube.data[i - n:i + n, j - n:j + n, :, :], axis=(0, 1))
    datacube_average.crop_R((n, datacube.R_Nx - n, n, datacube.R_Ny - n))
    # store the average information into an array
    datacube_info = np.zeros(
        (datacube_average.data.shape[0], datacube_average.data.shape[0], datacube.Q_Nx, datacube.Q_Ny))
    for i in range(datacube.R_Nx):
        for j in range(datacube.R_Ny):
            datacube_average.data[i][j] = avg[i, j, :, :]
            datacube_info[i][j] = datacube.data[i, j, :, :]
    return datacube_average, datacube_info


def make_Fourier_coords2D(Nx, Ny, pixelSize=1):
    """
        Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
        """
    if hasattr(pixelSize, '__len__'):
        assert len(
            pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_x = pixelSize[0]
        pixelSize_y = pixelSize[1]
    else:
        pixelSize_x = pixelSize
        pixelSize_y = pixelSize

    qx = np.fft.fftfreq(Nx, pixelSize_x)
    qy = np.fft.fftfreq(Ny, pixelSize_y)
    qy, qx = np.meshgrid(qy, qx)
    return qx, qy


def get_shifted_ar(
    ar,
    xshift,
    yshift,
    periodic=True,
    bilinear=False,
):
    "Shift function from py4DSTEM package."
    if bilinear is False:
        nx, ny = np.shape(ar)
        qx, qy = make_Fourier_coords2D(nx, ny, 1)
        nx, ny = float(nx), float(ny)

        w = np.exp(-(2j * np.pi) * ((yshift * qy) + (xshift * qx)))
        shifted_ar = np.real(np.fft.ifft2((np.fft.fft2(ar)) * w))

    else:
        xF = (np.floor(xshift)).astype(int)
        yF = (np.floor(yshift)).astype(int)
        wx = xshift - xF
        wy = yshift - yF

        shifted_ar = \
            np.roll(ar, (xF, yF), axis=(0, 1)) * ((1 - wx) * (1 - wy)) + \
            np.roll(ar, (xF + 1, yF), axis=(0, 1)) * ((wx) * (1 - wy)) + \
            np.roll(ar, (xF, yF + 1), axis=(0, 1)) * ((1 - wx) * (wy)) + \
            np.roll(ar, (xF + 1, yF + 1), axis=(0, 1)) * ((wx) * (wy))

    if periodic is False:
        # Rounded coordinates for boundaries
        xR = (np.round(xshift)).astype(int)
        yR = (np.round(yshift)).astype(int)

        if xR > 0:
            shifted_ar[0:xR, :] = 0
        elif xR < 0:
            shifted_ar[xR:, :] = 0
        if yR > 0:
            shifted_ar[:, 0:yR] = 0
        elif yR < 0:
            shifted_ar[:, yR:] = 0

    return shifted_ar


def symmetrize_datacube(datacube_average, probe_kernel_FT, origin_x, origin_y):
    """
    Parameters
    ----------
    datacube_average : TYPE
        DESCRIPTION.
    probe_kernel_FT : TYPE
        DESCRIPTION.
    origin_x : i_0.
    origin_y : j_0.

    Returns
    -------
    datacube_symmetrize : a datacube object
    datacube_info_s : a numpy array object (for visualization)
    """
    datacube_symmetrize = datacube_average
    # calculate the area that could be symmetrized
    sym_x = min(origin_x, datacube_average.Q_Nx - 1 - origin_x)
    sym_y = min(origin_y, datacube_average.Q_Ny - 1 - origin_y)
    # calculate the four points of the area
    up_left, bottom_left = (max(0, origin_x - sym_x), max(0, origin_y - sym_y)
                            ), (min(origin_x + sym_x, datacube_average.Q_Nx), max(0, origin_y - sym_y))
    #up_right,bottem_right=(max(0,origin_x-sym_x),min(datacube_average.Q_Ny,origin_y+sym_y)), (min(origin_x+sym_x,datacube_average.Q_Nx),min(datacube_average.Q_Ny,origin_y+sym_y))
    # generate the index list in diffration space, only need half
    index_ls = []
    for i in range(datacube_average.Q_Nx * datacube_average.Q_Ny):
        Qx, Qy = np.unravel_index(i, datacube_average.data.shape[2:])
        if (Qx >= up_left[0]) & (Qx < bottom_left[0]) & (Qy >= up_left[1]) & (Qy <= origin_y):
            index_ls.append(i)
    # first recenter each DP at [origin_x, origin_y]
    for n in py4DSTEM.process.utils.tqdmnd(range(datacube_average.R_N)):
        Rx, Ry = np.unravel_index(n, datacube_average.data.shape[:2])
        intensity = py4DSTEM.process.utils.get_cross_correlation_FT(
            datacube_average.data[Rx, Ry], probe_kernel_FT, corrPower=1)
        orig_intensity = intensity[origin_x, origin_y]
        curr_orig_x, curr_orig_y = origin_x, origin_y
        for w_x in range(origin_x - 4, origin_x + 4):
            for w_y in range(origin_y - 4, origin_y + 4):
                if intensity[w_x, w_y] > orig_intensity:
                    curr_orig_x, curr_orig_y = w_x, w_y
                    orig_intensity = intensity[w_x, w_y]
        shift_x, shift_y = origin_x - curr_orig_x, origin_y - curr_orig_y
        datacube_symmetrize.data[Rx][Ry] = get_shifted_ar(
            datacube_average.data[Rx][Ry], shift_x, shift_y, bilinear=True, periodic=False)
        for j in range(len(index_ls)):
            Qx, Qy = np.unravel_index(
                index_ls[j], datacube_average.data.shape[2:])
            Qx_s, Qy_s = 2 * origin_x - Qx, 2 * origin_y - Qy
            datacube_symmetrize.data[Rx, Ry, Qx, Qy] = datacube_symmetrize.data[Rx, Ry, Qx_s, Qy_s] = max(
                datacube_symmetrize.data[Rx, Ry, Qx, Qy], datacube_symmetrize.data[Rx, Ry, Qx_s, Qy_s])
    # store the symmetrization information into an array
    datacube_info_s = np.zeros(
        (datacube_symmetrize.data.shape[0], datacube_symmetrize.data.shape[0], datacube_symmetrize.Q_Nx, datacube_symmetrize.Q_Ny))
    for i in range(datacube_symmetrize.R_Nx):
        for j in range(datacube_symmetrize.R_Ny):
            datacube_info_s[i, j, :, :] = datacube_symmetrize.data[i, j, :, :]
    return datacube_symmetrize, datacube_info_s


def IsParticle(data, thres=10000):
    # datacube: 4D diffraction data
    # first 2 axes are the real image spatial location
    # last 2 axes are the diffraction intensity
    datasum = np.sum(data, axis=-1)
    datasum = np.sum(datasum, axis=-1)

    thres = thres
    datapar = datasum.copy()
    datapar[datasum <= thres] = 1
    datapar[datasum > thres] = 0
    return datapar


def generate_particle_index(datacube, particle_labels):
    '''
    Input: datacube, the 0/1 labels from particle identification
    Output: list of idx of pixels that belong/not belong to particle area, also the counts of particle index
    '''
    particle_index = []
    nonparticle_index = []
    q_hei = datacube.data.shape[:2]
    for n in py4DSTEM.process.utils.tqdmnd(range(datacube.R_N)):
        i, j = np.unravel_index(n, q_hei)
        if particle_labels[i][j] == 1:
            # particle_index.append(i*datacube.R_Nx+j)
            particle_index.append((i, j))
        else:
            nonparticle_index.append((i, j))
    particle_count = len(particle_index)
    return particle_index, nonparticle_index, particle_count


def bg_param(datacube_symmetrize, nonparticle_index, origin_x, origin_y):
    """Obtain amplitude parameters from pixels in the background
    """
    rad, intensity = radial_curve(
        datacube_symmetrize.data[0, 0], center=[origin_x, origin_y])
    average_intensity = np.zeros(len(intensity))
    for n in py4DSTEM.process.utils.tqdmnd(range(len(nonparticle_index))):
        Rx, Ry = nonparticle_index[n][0], nonparticle_index[n][1]
        rad, intensity = radial_curve(
            datacube_symmetrize.data[Rx, Ry], center=[origin_x, origin_y])
        intensity[:int((6 - 1) / 0.25)] = 0
        average_intensity += intensity
    average_intensity = average_intensity / len(nonparticle_index)

    def func(x, a, b, c, d, e):
        r_1 = np.ones((x.shape[0],)) * d
        fit_y = a * np.exp(-b * (x - r_0)) + c * \
            np.exp(-(x - r_1)**2 / (2 * e**2))
        return fit_y

    max_index = np.argmax(average_intensity)
    min_index = int((15 - 1) / 0.25)
    x = rad[max_index:min_index + 1]
    y = average_intensity[max_index:min_index + 1]
    r_0 = np.ones((x.shape[0],)) * (max_index * 0.25 + 1)
    alpha_bg = optimize.curve_fit(func, xdata=x, ydata=y, bounds=(
        [0, 0, 0, 8, 0], [30, 1, 15, 14, 9]), p0=[1, 1, 1, 11, 1], maxfev=20000)[0]
    # draw the plot
    a, b, c, d, e = alpha_bg[0], alpha_bg[1], alpha_bg[2], alpha_bg[3], alpha_bg[4]
    # fitting part
    x = rad[max_index:min_index + 1]
    r_0 = np.ones((x.shape[0],)) * (max_index * 0.25 + 1)
    r_1 = np.ones((x.shape[0],)) * d
    fit_y = a * np.exp(-b * (x - r_0)) + c * np.exp(-(x - r_1)**2 / (2 * e**2))
    # extrapolation part
    x_extra = rad[min_index:rad.shape[0]]
    r_0 = np.ones((x_extra.shape[0],)) * (max_index * 0.25 + 1)
    r_1 = np.ones((x_extra.shape[0],)) * d
    y_extra = a * np.exp(-b * (x_extra - r_0)) + c * \
        np.exp(-(x_extra - r_1)**2 / (2 * e**2))
    # deduct fitting/extrapolation part
    x_deduce = rad[max_index:len(average_intensity)]
    y_deduce = np.zeros(x_deduce.shape[0])
    for i in range(max_index, min_index + 1):
        if average_intensity[i] > fit_y[i - max_index]:
            y_deduce[i - max_index] = average_intensity[i] - \
                fit_y[i - max_index]
        else:
            y_deduce[i - max_index] = 0
    for i in range(min_index + 1, len(intensity)):
        if average_intensity[i] > y_extra[i - (min_index + 1)]:
            y_deduce[i - max_index] = average_intensity[i] - \
                y_extra[i - (min_index + 1)]
        else:
            y_deduce[i - max_index] = 0
    return alpha_bg


def particle_param(datacube_symmetrize, particle_index, alpha_bg, origin_x, origin_y):
    """Obtain location parameters for pixels in the particle and store as a list
    """
    par_para = np.zeros((len(particle_index), 2))
    par_error = [None]
    for n in py4DSTEM.process.utils.tqdmnd(range(len(particle_index))):
        Rx, Ry = particle_index[n][0], particle_index[n][1]
        rad, intensity = radial_curve(
            datacube_symmetrize.data[Rx, Ry], center=[origin_x, origin_y])
        intensity[:int((6 - 1) / 0.25)] = 0
        max_index = np.argmax(intensity[:int((10 - 1) / 0.25)])
        min_index = int((15 - 1) / 0.25)
        x = rad[max_index:min_index + 1]
        y = intensity[max_index:min_index + 1]
        r_0 = np.ones((x.shape[0],)) * (max_index * 0.25 + 1)
        #r_1 = np.ones((x.shape[0],)) * (min_index*0.25+1)

        def func(x, a, b):
            r_1 = np.ones((x.shape[0],)) * alpha_bg[3]
            fit_y = a * (np.exp(-alpha_bg[1] * (x - r_0))) + \
                b * np.exp(-(x - r_1)**2 / (2 * alpha_bg[4]**2))
            return fit_y
        try:
            alpha = optimize.curve_fit(func, xdata=x, ydata=y, bounds=(
                [0, 0], [30, 30]), p0=[1, 1], maxfev=20000)[0]
            par_para[n] = alpha
        except:
            par_para[n] = par_para[n - 1]
            par_error.append(n)
            pass
    return par_para


def particle_curve(datacube_symmetrize, datacube_info_s, Rx, Ry, particle_index, par_para, alpha_bg, origin_x, origin_y):
    """Plot of curves for original radial integration, the fitting curve, and the residue
    """
    index = particle_index.index((Rx, Ry))
    a, b = par_para[index]
    rad, intensity = radial_curve(
        datacube_info_s[Rx, Ry], center=[origin_x, origin_y])
    intensity[:int((6 - 1) / 0.25)] = 0
    #r_0 = np.ones((x_extra.shape[0],)) * (max_index*0.25+1)
    # fitting part
    max_index = np.argmax(intensity[:int((10 - 1) / 0.25)])
    min_index = int((15 - 1) / 0.25)
    x = rad[max_index:min_index + 1]
    r_0 = np.ones((x.shape[0],)) * (max_index * 0.25 + 1)
    r_1 = np.ones((x.shape[0],)) * alpha_bg[3]
    fit_y = a * np.exp(-alpha_bg[1] * (x - r_0)) + \
        b * np.exp(-(x - r_1)**2 / (2 * alpha_bg[4]**2))
    # extrapolation part
    x_extra = rad[min_index:rad.shape[0]]
    r_0 = np.ones((x_extra.shape[0],)) * (max_index * 0.25 + 1)
    r_1 = np.ones((x_extra.shape[0],)) * alpha_bg[3]
    y_extra = a * np.exp(-alpha_bg[1] * (x_extra - r_0)) + \
        b * np.exp(-(x_extra - r_1)**2 / (2 * alpha_bg[4]**2))
    # deduct fitting/extrapolation
    x_deduce = rad[max_index:len(intensity)]
    y_deduce = np.zeros(x_deduce.shape[0])
    for i in range(max_index, min_index + 1):
        if intensity[i] > fit_y[i - max_index]:
            y_deduce[i - max_index] = intensity[i] - fit_y[i - max_index]
        else:
            y_deduce[i - max_index] = 0
    for i in range(min_index + 1, len(intensity)):
        if intensity[i] > y_extra[i - (min_index + 1)]:
            y_deduce[i - max_index] = intensity[i] - \
                y_extra[i - (min_index + 1)]
        else:
            y_deduce[i - max_index] = 0
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot data
    ax.plot(rad, intensity, linewidth=2, label='raw')
    ax.plot(rad[max_index:min_index + 1], fit_y, linewidth=2, label='fitted')
    ax.plot(x_extra, y_extra, linewidth=2, label='extrapolation')
    ax.plot(x_deduce, y_deduce, linewidth=2, label='residual')
    ax.set_xlabel("radius")
    ax.set_ylabel("pixel intensity")
    ax.legend()
    ax.set_xlim(1, 50)


def subtract_bg(datacube_symmetrize, particle_index, par_param, center, alpha_bg):
    """Bg subtraction for all pixels in the particle
    """
    def subtract_bg_DP(datacube_symmetrize, Rx, Ry, par_param, center, alpha_bg):
        a, b = par_param

        data = np.copy(datacube_symmetrize.data[Rx, Ry])
        for radius in idx_mp.keys():
            if radius < 6:
                for (i, j) in idx_mp[radius]:
                    data[i, j] = 0
            else:
                subtraction = a * \
                    math.exp(-alpha_bg[1] * (radius - 6)) + b * \
                    np.exp(-(radius - alpha_bg[3])**2 / constant)
                for (i, j) in idx_mp[radius]:
                    data[i, j] = max(0, data[i, j] - subtraction)
        return data
    # bg subtraction for every pixel
    row, col = datacube_symmetrize.Q_Nx, datacube_symmetrize.Q_Ny
    X, Y = np.meshgrid(np.arange(row), np.arange(col))
    r = np.sqrt(np.square(X - center[1]) + np.square(Y - center[0]))
    idx_mp = {}
    for i in range(row):
        for j in range(col):
            if r[i][j] not in idx_mp:
                idx_mp[r[i][j]] = [(i, j)]
            else:
                idx_mp[r[i][j]].append((i, j))

    constant = 2 * alpha_bg[4]**2
    for n in py4DSTEM.process.utils.tqdmnd(range(len(particle_index))):
        Rx, Ry = particle_index[n][0], particle_index[n][1]
        data = subtract_bg_DP(datacube_symmetrize, Rx, Ry,
                              par_param[n], constant, alpha_bg)
        datacube_symmetrize.data[Rx, Ry, :, :] = data[:, :]

    return datacube_symmetrize


def get_bragg_vector_map_raw(braggpeaks, Q_Nx, Q_Ny, R_Nx, R_Ny):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, where
    the peak positions have not been centered.
    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels
    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)): the bragg vector map
    """
    qx = []
    qy = []
    I = []
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            qx.extend(braggpeaks.raw[Rx, Ry]._data['qx'])
            qy.extend(braggpeaks.raw[Rx, Ry]._data['qy'])
            I.extend(braggpeaks.raw[Rx, Ry]._data['intensity'])
    qx = np.array(qx)
    qy = np.array(qy)
    I = np.array(I)

    # Precompute rounded coordinates
    floorx = np.floor(qx).astype(np.int64)
    ceilx = np.ceil(qx).astype(np.int64)
    floory = np.floor(qy).astype(np.int64)
    ceily = np.ceil(qy).astype(np.int64)

    # Remove any points outside [0, Q_Nx] & [0, Q_Ny]
    mask = np.logical_and.reduce(
        ((floorx >= 0), (floory >= 0), (ceilx < Q_Nx), (ceily < Q_Ny)))
    qx = qx[mask]
    qy = qy[mask]
    I = I[mask]
    floorx = floorx[mask]
    floory = floory[mask]
    ceilx = ceilx[mask]
    ceily = ceily[mask]

    dx = qx - floorx
    dy = qy - floory

    # Compute indices of the 4 neighbors to (qx,qy)
    # floor x, floor y
    inds00 = np.ravel_multi_index([floorx, floory], (Q_Nx, Q_Ny))
    # floor x, ceil y
    inds01 = np.ravel_multi_index([floorx, ceily], (Q_Nx, Q_Ny))
    # ceil x, floor y
    inds10 = np.ravel_multi_index([ceilx, floory], (Q_Nx, Q_Ny))
    # ceil x, ceil y
    inds11 = np.ravel_multi_index([ceilx, ceily], (Q_Nx, Q_Ny))

    # Compute the BVM by accumulating intensity in each neighbor weighted by linear interpolation
    bvm = (np.bincount(inds00, I * (1. - dx) * (1. - dy), minlength=Q_Nx * Q_Ny) +
           np.bincount(inds01, I * (1. - dx) * dy, minlength=Q_Nx * Q_Ny) +
           np.bincount(inds10, I * dx * (1. - dy), minlength=Q_Nx * Q_Ny) +
           np.bincount(inds11, I * dx * dy, minlength=Q_Nx * Q_Ny)).reshape(Q_Nx, Q_Ny)

    return bvm


def ObtainIntensityMatrix(datacube, braggpeaks, probe_kernel_FT, Qx, Qy, max_dist=None):
    """
    Create intensity matrix I for the nanoparticle. Return a 3_D matrix, and the last dimension
    records the feature vector for each scan position.
    """
    R_Nx = braggpeaks.shape[0]  #: shape of real space (x)
    R_Ny = braggpeaks.shape[1]  #: shape of real space (y)
    #: the sets of Bragg peaks present at each scan position
    braggpeak_labels = get_braggpeak_labels_by_scan_position(braggpeaks,
                                                             R_Nx, R_Ny, Qx, Qy, max_dist)

    # Construct X matrix
    N_feat = len(Qx)  # the number of bragg peaks
    X = np.zeros((R_Nx, R_Ny, N_feat))
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            s = braggpeak_labels[Rx][Ry]
            pointlist = braggpeaks.raw[Rx, Ry]
            for i in s:
                ind = np.argmin(np.hypot(pointlist._data['qx'] - Qx[i],
                                         pointlist._data['qy'] - Qy[i]))
                X[Rx, Ry, i] = pointlist._data['intensity'][ind]
            cc_intensity = py4DSTEM.process.utils.get_cross_correlation_FT(
                datacube.data[Rx, Ry], probe_kernel_FT, corrPower=1)
            not_s = [i for i in range(N_feat) if i not in s]
            for i in not_s:
                X[Rx, Ry, i] = max(cc_intensity[round(Qx[i]), round(Qy[i])], 0)
    return X


def get_braggpeak_labels_by_scan_position(braggpeaks, R_Nx, R_Ny, Qx, Qy, max_dist=None):
    braggpeak_labels = [[set() for i in range(braggpeaks.shape[1])]
                        for j in range(braggpeaks.shape[0])]
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            s = braggpeak_labels[Rx][Ry]
            pointlist = braggpeaks.raw[Rx, Ry]
            for i in range(pointlist.data.shape[0]):
                label = np.argmin(
                    np.hypot(Qx - pointlist._data['qx'][i], Qy - pointlist._data['qy'][i]))
                if max_dist is not None:
                    if np.hypot(Qx[label] - pointlist._data['qx'][i], Qy[label] - pointlist._data['qy'][i]) < max_dist:
                        s.add(label)
                else:
                    s.add(label)

    return braggpeak_labels


def find_beta(datacube, braggpeaks_prob, particle_labels):
    index = np.array(np.where(particle_labels == 1)).T
    n = index.shape[0]
    indexinfo = [[np.linalg.norm(index[i] - index[j], ord=None),
                  np.linalg.norm(braggpeaks_prob[tuple(index[i])] - braggpeaks_prob[tuple(index[j])], ord=None)]
                 for i in range(n) for j in range(n) if i < j]
    mp = {}
    for d_real, d_reci in indexinfo:
        if d_real not in mp:
            mp[d_real] = (d_reci, 1)
        else:
            ave_reci, count = mp[d_real]
            ave_reci = (ave_reci * count + d_reci) / (count + 1)
            mp[d_real] = (ave_reci, count + 1)
    storex = np.array([key for key in mp.keys()])
    storey = np.array([ave_reci for ave_reci, count in mp.values()])
    print(indexinfo[:5], storex[:5], storey[:5])
    storexsquare, storeysquare = np.square(storex), np.square(storey)
    r = np.mean(storeysquare) / np.mean(storexsquare)
    beta = np.sqrt(4.92e-8 / r)
    return beta


def plot_transit_prob(gm, cov_ls, color_ls, pca, braggpeaks_intensity, regu_para,
                      beta, label, cluster1, cluster2, n):
    cluster1 -= 1
    cluster2 -= 1

    mean = gm.means_
    cov_ls = cov_ls

    mean1 = gm.means_[cluster1, :2]
    mean2 = gm.means_[cluster2, :2]

    dist = np.sqrt((mean2[0] - mean1[0])**2 + (mean2[1] - mean1[1])**2)
    dirc = np.array([mean2[0] - mean1[0], mean2[1] - mean1[1]]) / dist

    pos = mean1 + np.array([i / n * dist * dirc for i in range(n)])
    idx_ls = []
    for i in range(pos.shape[0]):
        d = np.round(pos[i, 0])
        if not idx_ls or d != np.round(pos[idx_ls[-1], 0]):
            idx_ls.append(i)
    coord_ls = pos[idx_ls, ].astype(int)
    features_loc = braggpeaks_intensity[tuple(
        coord_ls[:, 0]), tuple(coord_ls[:, 1])]
    features_loc = pca.transform(features_loc)
    X_plot = np.hstack((coord_ls, beta * features_loc))

    n_samples = X_plot.shape[0]
    n_components, n_features = gm.means_.shape[0], gm.means_.shape[1]

    # for full matrix
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(cov_ls):
        try:
            cov_chol = linalg.cholesky(
                covariance + np.eye(n_features) * regu_para, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T

    log_det = np.sum(
        np.log(precisions_chol.reshape(n_components, -1)
               [:, :: n_features + 1]), 1
    )

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(mean, precisions_chol)):
        y = np.dot(X_plot - mu, prec_chol)
        log_prob[:, k] = np.sum(y**2, axis=1)

    prob_new = -0.5 * (n_features * np.log(2 * np.pi) +
                       log_prob) + log_det[np.newaxis, :]
    weighted_log_prob = prob_new + np.log(gm.weights_)
    log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    prob = np.exp(log_resp)

    middle_idx = (
        np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]) * n_samples).astype(int)
    middle_prob = prob[tuple(middle_idx), :]
    middle_points = coord_ls[list(middle_idx)].astype(int)

    return coord_ls, X_plot, prob, middle_idx, middle_points, middle_prob
