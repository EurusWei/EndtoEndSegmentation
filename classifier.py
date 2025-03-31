#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:20:16 2022

@author: weiliu
"""

from numpy.linalg import lstsq
from itertools import permutations
from scipy.ndimage import gaussian_filter
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
)
from skimage.measure import label
from sklearn.decomposition import NMF
import numpy as np
########################


class BraggVectorClassification(object):
    def __init__(self, braggpeaks, particle_labels, R_Nx, R_Ny, Qx, Qy, X_is_boolean=True, max_dist=None):
        assert len(Qx) == len(Qy), "Qx and Qy must have same length"
        self.braggpeaks = braggpeaks
        self.R_Nx = R_Nx  #: shape of real space (x)
        self.R_Ny = R_Ny  #: shape of real space (y)
        self.Qx = Qx  #: x-coordinates of the voronoi points
        self.Qy = Qy  #: y-coordinates of the voronoi points

        #: the sets of Bragg peaks present at each scan position
        self.braggpeak_labels = get_braggpeak_labels_by_scan_position(
            braggpeaks, R_Nx, R_Ny, Qx, Qy, max_dist
        )

        # Construct X matrix
        #: first dimension of the data matrix; the number of bragg peaks
        self.N_feat = len(self.Qx)
        #: second dimension of the data matrix; the number of scan positions
        self.N_meas = self.R_Nx * self.R_Ny

        x = np.zeros((self.R_Nx, self.R_Ny, self.N_feat))  #: the data matrix
        for Rx in range(self.R_Nx):
            for Ry in range(self.R_Ny):
                #/**/
                if particle_labels[Rx][Ry] == 1:
                    s = self.braggpeak_labels[Rx][Ry]
                    pointlist = self.braggpeaks.pointlists[Rx][Ry].data
                    for i in s:
                        if X_is_boolean:
                            x[Rx, Ry, i] = True
                        else:
                            ind = np.argmin(
                                np.hypot(
                                    pointlist["qx"] - Qx[i],
                                    pointlist["qy"] - Qy[i],
                                )
                            )
                            x[Rx, Ry, i] = pointlist["intensity"][ind]
        self.X = x[particle_labels == 1].T

        return

    def get_initial_classes_by_cooccurrence(
        self,
        thresh=0.3,
        BP_fraction_thresh=0.1,
        max_iterations=200,
        X_is_boolean=True,
        n_corr_init=2,
    ):

        assert isinstance(X_is_boolean, bool)
        assert isinstance(max_iterations, (int, np.integer))
        assert n_corr_init in (2, 3)

        # Get sets of integers representing the initial classes
        BP_sets = get_initial_classes(
            self.braggpeak_labels,
            N=len(self.Qx),
            thresh=thresh,
            BP_fraction_thresh=BP_fraction_thresh,
            max_iterations=max_iterations,
            n_corr_init=n_corr_init,
        )

        # Construct W, H matrices
        self.N_c = len(BP_sets)

        # W
        self.W = np.zeros((self.N_feat, self.N_c))
        for i in range(self.N_c):
            BP_set = BP_sets[i]
            for j in BP_set:
                self.W[j, i] = 1

        # H
        self.H = lstsq(self.W, self.X, rcond=None)[0]
        self.H = np.where(self.H < 0, 0, self.H)

        self.W_next = None
        self.H_next = None
        self.N_c_next = None

        return

    def nmf(self, max_iterations=1):
        sklearn_nmf = NMF(n_components=self.N_c, init="custom", max_iter=max_iterations)
        self.W_next = sklearn_nmf.fit_transform(self.X, W=self.W, H=self.H)
        self.H_next = sklearn_nmf.components_
        self.N_c_next = self.W_next.shape[1]

        return

### Functions for initial class determination ###


def get_braggpeak_labels_by_scan_position(braggpeaks, R_Nx, R_Ny, Qx, Qy, max_dist=None):
    """
    For each scan position, gets a set of integers, specifying the bragg peaks at this
    scan position.

    From a set of positions in diffraction space (Qx,Qy), assign each detected bragg peak
    in the PointListArray braggpeaks a label corresponding to the index of the closest
    position; thus for a bragg peak at (qx,qy), if the closest position in (Qx,Qy) is
    (Qx[i],Qy[i]), assign this peak the label i. This is equivalent to assigning each
    bragg peak (qx,qy) a label according to the Voronoi region it lives in, given a
    voronoi tesselation seeded from the points (Qx,Qy).

    For each scan position, get the set of all indices i for all bragg peaks found at
    this scan position.

    Args:
        braggpeaks (PointListArray): Bragg peaks; must have coords 'qx' and 'qy'
        Qx (ndarray of floats): x-coords of the voronoi points
        Qy (ndarray of floats): y-coords of the voronoi points
        max_dist (None or number): maximum distance from a given voronoi point a peak
            can be and still be associated with this label

    Returns:
        (list of lists of sets) the labels found at each scan position. Scan position
        (Rx,Ry) is accessed via braggpeak_labels[Rx][Ry]
    """
    braggpeak_labels = [
        [set() for i in range(R_Nx)] for j in range(R_Ny)
    ]
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            s = braggpeak_labels[Rx][Ry]
            pointlist = braggpeaks.pointlists[Rx][Ry].data
            for i in range(pointlist.shape[0]):
                label = np.argmin(
                    np.hypot(Qx - pointlist['qx'][i], Qy - pointlist["qy"][i])
                )
                if max_dist is not None:
                    if (
                        np.hypot(
                            Qx[label] - pointlist["qx"][i],
                            Qy[label] - pointlist["qy"][i],
                        )
                        < max_dist
                    ):
                        s.add(label)
                else:
                    s.add(label)

    return braggpeak_labels


def get_initial_classes(
    braggpeak_labels,
    N,
    thresh=0.3,
    BP_fraction_thresh=0.1,
    max_iterations=200,
    n_corr_init=2,
):
    """
    From the sets of Bragg peaks present at each scan position, get an initial guess
    classes at which Bragg peaks should be grouped together into classes.

    The algorithm is as follows:
    1. Calculate an n-point correlation function, i.e. the joint probability of any given
    n BPs coexisting in a diffraction pattern.  n is controlled by n_corr_init, and must
    be 2 or 3. peaks i, j, and k are all in the same DP.
    2. Find the BP triplet maximizing the 3-point function; include these three BPs in a
    class.
    3. Get all DPs containing the class BPs. From these, find the next most likely BP to
    also be present.  If its probability of coexisting with the known class BPs is
    greater than thresh, add it to the class and repeat this step. Otherwise, proceed to
    the next step.
    4. Check: if the new class is the same as a class that has already been found, OR if
    the fraction of BPs which have not yet been placed in a class is less than
    BP_fraction_thresh, or more than max_iterations have been attempted, finish,
    returning all classes. Otherwise, set all slices of the 3-point function containing
    the BPs in the new class to zero, and begin a new iteration, starting at step 2 using
    the new, altered 3-point function.

    Args:
        N (int): the total number of indexed Bragg peaks in the 4D-STEM dataset
        braggpeak_labels (list of lists of sets): the Bragg peak labels found at each
            scan position; see get_braggpeak_labels_by_scan_position().
        thresh (float in [0,1]): threshold for adding new BPs to a class
        BP_fraction_thresh (float in [0,1]): algorithm terminates if fewer than this
            fraction of the BPs have not been assigned to a class
        max_iterations (int): algorithm terminates after this many iterations
        n_corr_init (int): seed new classes by finding maxima of the n-point joint
            probability function.  Must be 2 or 3.

    Returns:
        (list of sets): the sets of Bragg peaks constituting the classes
    """
    assert isinstance(braggpeak_labels[0][0], set)
    assert thresh >= 0 and thresh <= 1
    assert BP_fraction_thresh >= 0 and BP_fraction_thresh <= 1
    assert isinstance(max_iterations, (int, np.integer))
    assert n_corr_init in (2, 3)
    R_Nx = len(braggpeak_labels)
    R_Ny = len(braggpeak_labels[0])

    if n_corr_init == 2:
        # Get two-point function
        n_point_function = np.zeros((N, N))
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                s = braggpeak_labels[Rx][Ry]
                perms = permutations(s, 2)
                for perm in perms:
                    n_point_function[perm[0], perm[1]] += 1
        n_point_function /= R_Nx * R_Ny

        # Main loop
        BP_sets = []
        iteration = 0
        unused_BPs = np.ones(N, dtype=bool)
        seed_new_class = True
        while seed_new_class:
            ind1, ind2 = np.unravel_index(np.argmax(n_point_function), (N, N))
            BP_set = set([ind1, ind2])
            grow_class = True
            while grow_class:
                frequencies = np.zeros(N)
                N_elements = 0
                for Rx in range(R_Nx):
                    for Ry in range(R_Ny):
                        s = braggpeak_labels[Rx][Ry]
                        if BP_set.issubset(s):
                            N_elements += 1
                            for i in s:
                                frequencies[i] += 1
                frequencies /= N_elements
                for i in BP_set:
                    frequencies[i] = 0
                ind_new = np.argmax(frequencies)
                if frequencies[ind_new] > thresh:
                    BP_set.add(ind_new)
                else:
                    grow_class = False

            # Modify 2-point function, add new BP set to list, and decide to continue or stop
            for i in BP_set:
                n_point_function[i, :] = 0
                n_point_function[:, i] = 0
                unused_BPs[i] = 0
            for s in BP_sets:
                if len(s) == len(s.union(BP_set)):
                    seed_new_class = False
            if seed_new_class is True:
                BP_sets.append(BP_set)
            iteration += 1
            N_unused_BPs = np.sum(unused_BPs)
            if iteration > max_iterations or N_unused_BPs < N * BP_fraction_thresh:
                seed_new_class = False

    else:
        # Get three-point function
        n_point_function = np.zeros((N, N, N))
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                s = braggpeak_labels[Rx][Ry]
                perms = permutations(s, 3)
                for perm in perms:
                    n_point_function[perm[0], perm[1], perm[2]] += 1
        n_point_function /= R_Nx * R_Ny

        # Main loop
        BP_sets = []
        iteration = 0
        unused_BPs = np.ones(N, dtype=bool)
        seed_new_class = True
        while seed_new_class:
            ind1, ind2, ind3 = np.unravel_index(
                np.argmax(n_point_function), (N, N, N))
            BP_set = set([ind1, ind2, ind3])
            grow_class = True
            while grow_class:
                frequencies = np.zeros(N)
                N_elements = 0
                for Rx in range(R_Nx):
                    for Ry in range(R_Ny):
                        s = braggpeak_labels[Rx][Ry]
                        if BP_set.issubset(s):
                            N_elements += 1
                            for i in s:
                                frequencies[i] += 1
                frequencies /= N_elements
                for i in BP_set:
                    frequencies[i] = 0
                ind_new = np.argmax(frequencies)
                if frequencies[ind_new] > thresh:
                    BP_set.add(ind_new)
                else:
                    grow_class = False

            # Modify 3-point function, add new BP set to list, and decide to continue or stop
            for i in BP_set:
                n_point_function[i, :, :] = 0
                n_point_function[:, i, :] = 0
                n_point_function[:, :, i] = 0
                unused_BPs[i] = 0
            for s in BP_sets:
                if len(s) == len(s.union(BP_set)):
                    seed_new_class = False
            if seed_new_class is True:
                BP_sets.append(BP_set)
            iteration += 1
            N_unused_BPs = np.sum(unused_BPs)
            if iteration > max_iterations or N_unused_BPs < N * BP_fraction_thresh:
                seed_new_class = False

    return BP_sets
