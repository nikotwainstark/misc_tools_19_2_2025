import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import pandas as pd
from matplotlib.widgets import Slider

module_path = r"C:\GitHub\new pyir\PyIR\src"
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append(r"C:\PythonProjects\PhD Project\GitHub_test")
import smartimage
from pyir_spectralcollection import PyIR_SpectralCollection
path = r"Z:\Masters and UG Projects\2021-2022\Dilara and Amelia\PR484_C051\PR484_C051_e2\PR484_c051_e2.dmt"
tile = PyIR_SpectralCollection(path)
tile.is_loaded()

tissue_mask = tile.area_between(1600,1700, tile.data, tile.wavenumbers) >= 2

tissue_data = tile.data[tissue_mask]
tissue_data = tile.min2zero(tissue_data)
tissue_data, wavn = tile.remove_wax(tissue_data, tile.wavenumbers)
tissue_data, wavn = tile.keep_range(900,1800, tissue_data, tile.wavenumbers)
tissue_data = tile.vector_norm(tissue_data)

mean_centered_data = tissue_data - np.mean(tissue_data, axis=0)
image_rebulid = np.zeros_like(tile.data[:,:tissue_data.shape[1]])
image_rebulid[tissue_mask, :] = tissue_data
# image_3d = image_rebulid.reshape(tile.ypixels, tile.xpixels, -1)

# test_difference = image_rebulid[:-1, :] - image_rebulid[1:, :]
# # difference_2d = test_difference.reshape(-1, image_rebulid.shape[1])
# # difference_cov = np.cov(difference_2d, rowvar=False)
# difference_cov = np.cov(test_difference, rowvar=False)
# sample_cov = np.cov(tissue_data, rowvar=False)
# sample_cov_inverse = np.linalg.inv(sample_cov + 1e-6 * np.eye(sample_cov.shape[0]))
# noise_signal_cov = np.dot(difference_cov, sample_cov_inverse)
# eigvalues, eigvectors = np.linalg.eig(noise_signal_cov)
# sorted_indices = np.argsort(eigvalues)
# sorted_eigvalue = eigvalues[sorted_indices]
# sorted_eigvector = eigvectors[:, sorted_indices]
# linear_transform = sorted_eigvector[: , :20]
# denoised_subspace = np.dot(linear_transform.T, tissue_data.T)
# denoised_data = np.dot(linear_transform, denoised_subspace).T
# denoised_data = denoised_data + np.mean(tissue_data, axis=0)


# Calculate the difference covariance for noise estimation
test_difference = image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[:,:-1,:] - image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[:,1:,:]
difference_cov = np.cov(test_difference, rowvar=False)

# Sample covariance matrix and its inverse
sample_cov = np.cov(tissue_data, rowvar=False)
sample_cov_inverse = np.linalg.inv(sample_cov + 1e-6 * np.eye(sample_cov.shape[0]))

# Calculate the noise-signal covariance matrix
noise_signal_cov = np.dot(difference_cov, sample_cov_inverse)

# Eigen decomposition and sorting in descending order
eigvalues, eigvectors = np.linalg.eig(noise_signal_cov)
sorted_indices = np.argsort(eigvalues)[::-1]  # Descending order
sorted_eigvalue = eigvalues[sorted_indices]
sorted_eigvector = eigvectors[:, sorted_indices]

# Determine number of components based on threshold criterion
S2_diag = sorted_eigvalue - 1
K = np.sum(S2_diag > 5.0)  # Retain only high signal-to-noise components
linear_transform = sorted_eigvector[:, :K]

# Project to MNF subspace and reconstruct denoised data
denoised_subspace = np.dot(linear_transform.T, tissue_data.T)
denoised_data = np.dot(linear_transform, denoised_subspace).T

from sklearn.decomposition import PCA
pca = PCA(n_components=20)
denoised_data_pca = pca.fit_transform(tissue_data)
denoised_data_pca=pca.inverse_transform(denoised_data_pca)
plt.figure()
plt.plot(denoised_data_pca[50000, : ], c='red')
plt.plot(denoised_data[50000, :], c='blue')
plt.plot(tissue_data[50000,:], c='green')
plt.legend()
plt.show()


image_rebulid_denoised = np.zeros_like(image_rebulid)
image_rebulid_denoised[tissue_mask] = denoised_data
plt.figure()
denoised_1655 = image_rebulid_denoised[:, np.argmin(np.abs(wavn-1550))]
plt.imshow(denoised_1655.reshape(tile.ypixels, tile.xpixels))
plt.show()
plt.figure()
image_1655 = image_rebulid[:, np.argmin(np.abs(wavn-1700))]
plt.imshow(image_1655.reshape(tile.ypixels, tile.xpixels))
plt.show()

tile_denoised = PyIR_SpectralCollection()
tile_denoised.data = image_rebulid_denoised
tile_denoised.xpixels = tile.xpixels
tile_denoised.ypixels = tile.ypixels
tile_denoised.wavenumbers = wavn


import smartimage
si = smartimage.SmartImage(tile_denoised)
si.clickon()

# ————————————————————————————————————fast MNF————————————————————————————————————————————————————


def fast_mnf_denoise(hyperspectraldata, SNR = 5):
        """
        Perform Fast Minimum Noise Fraction (MNF) denoising on hyperspectral data.
        Code derived from supplementary information from Gupta et al:
        https://doi.org/10.1371/journal.pone.0205219

        This function reduces noise in hyperspectral images using the MNF
        transformation. The input data `hyperspectraldata` can be 2D or 3D.
        If the input is 3D (i.e., a hyperspectral image with spatial and
        spectral dimensions), it will be reshaped to 2D for processing and
        reshaped back to its original dimensions after denoising. If the input
        is already 2D, it will be processed directly.

        Steps:
            1. Compute the difference matrix `dX` for noise estimation.
            2. Perform eigenvalue decomposition on `dX^T * dX`.
            3. Weight the input data by the inverse square root of the eigenvalues.
            4. Perform eigenvalue decomposition on the weighted data.
            5. Retain the top K components based on Rose's noise criterion.
            6. Compute the transformation matrices `Phi_hat` and `Phi_tilde`.
            7. Project the data onto MNF components and reconstruct the denoised data.

        Parameters
        ----------
        hyperspectraldata : numpy.ndarray
            The input hyperspectral data. Can be either a 2D array (pixels × spectral bands)
            or a 3D array (rows × columns × spectral bands).

        SNR : int
            The signal to noise ratio value for detectability threshold outlined
            in the Rose criterion for medical imaging signal detection theory.
            A SNR value of 5 (default)  orivudes a 95% probability of object
            detection by humans visually. This value can be changed under the
            assumption this function will be used in ML applications with less
            subjectivity.

        Returns
        -------
        clean_data : numpy.ndarray
            The denoised hyperspectral data. The output will have the same dimensions as the input:
            - If the input was 3D, the output will be reshaped back to 3D.
            - If the input was 2D, the output will remain 2D.

        Raises
        ------
        ValueError
            If the input array `C` is neither 2D nor 3D.

        Example
        -------
        >>> hyperspectraldata = np.random.rand(100, 100, 50)  # A 3D hyperspectral image
        >>> clean_data = fast_mnf_denoise(hyperspectraldata)
        >>> clean_data.shape
        (100, 100, 50)
        """

        # Check if the input is 3D and reshape to 2D if needed
        if hyperspectraldata.ndim == 3:
            m, n, s = hyperspectraldata.shape
            X = np.reshape(hyperspectraldata, (-1, s))  # Reshape to 2D
        elif hyperspectraldata.ndim == 2:
            X = hyperspectraldata
            m, n = X.shape
            s = n  # If already 2D, assume second dimension is the spectral dimension
        else:
            raise ValueError("Input C must be either 2D or 3D.")

        # Step 2: Create the dX matrix
        dX = np.zeros((m, s))
        for i in range(m - 1):
            dX[i, :] = X[i, :] - X[i + 1, :]

        # Step 3: Perform eigenvalue decomposition of dX' * dX
        S1, U1 = np.linalg.eigh(dX.T @ dX)
        ix = np.argsort(S1)[::-1]  # Sort in descending order
        U1 = U1[:, ix]
        D1 = S1[ix]
        diagS1 = 1.0 / np.sqrt(D1)

        # Step 4: Compute weighted X
        wX = X @ U1 @ np.diag(diagS1)

        # Step 5: Perform eigenvalue decomposition of wX' * wX
        S2, U2 = np.linalg.eigh(wX.T @ wX)
        iy = np.argsort(S2)[::-1]  # Sort in descending order
        U2 = U2[:, iy]
        D2 = S2[iy]

        # Step 6: Retain top K components according to input SNR threshold
        S2_diag = D2 - 1
        K = np.sum(S2_diag > SNR)
        U2 = U2[:, :K]

        # Step 7: Compute Phi_hat and Phi_tilde
        Phi_hat = U1 @ np.diag(diagS1) @ U2
        Phi_tilde = U1 @ np.diag(np.sqrt(D1)) @ U2

        # Step 8: Project data onto MNF components and reshape to original dimensions
        mnfX = X @ Phi_hat
        Xhat = mnfX @ Phi_tilde.T

        if hyperspectraldata.ndim == 3:
            clean_data = np.reshape(Xhat, (m, n, s))  # Reshape back to 3D if input was 3D
        else:
            clean_data = Xhat  # Keep 2D if input was 2D

        return clean_data

denoised_data_fast_mnf = fast_mnf_denoise(tissue_data, SNR=5)

plt.figure()
sg_smoothing, sgwavn1 = tile.data_deriv(tile.data,tile.wavenumbers,7,5,0)
sg_smoothing = tile.min2zero(sg_smoothing)
# sg_smoothing, sgwavn = tile.remove_wax(sg_smoothing, sgwavn)
# sg_smoothing, sgwavn = tile.keep_range(903,1800, sg_smoothing, sgwavn)
sg_smoothing = tile.vector_norm(sg_smoothing)
sg_smoothing2, sgwavn2 = tile.data_deriv(tile.data,tile.wavenumbers,7,5,1)
sg_smoothing2 = tile.min2zero(sg_smoothing2)
# sg_smoothing, sgwavn = tile.remove_wax(sg_smoothing, sgwavn)
# sg_smoothing, sgwavn = tile.keep_range(903,1800, sg_smoothing, sgwavn)
sg_smoothing2 = tile.vector_norm(sg_smoothing2)
sg_smoothing3, sgwavn3 = tile.data_deriv(tile.data,tile.wavenumbers,7,5,2)
sg_smoothing3 = tile.min2zero(sg_smoothing3)
# sg_smoothing, sgwavn = tile.remove_wax(sg_smoothing, sgwavn)
# sg_smoothing, sgwavn = tile.keep_range(903,1800, sg_smoothing, sgwavn)
sg_smoothing3 = tile.vector_norm(sg_smoothing3)
plt.plot(sgwavn1, sg_smoothing[50000,:]/100 +0.0260, c='red',label=f'Raw data')
plt.plot(sgwavn2, sg_smoothing2[50000,:]+0.002, c='blue', label=f'First Derivative')
plt.plot(sgwavn3, sg_smoothing3[50000,:],  c='green',label=f'Second Derivative')
plt.title('7-point spectral derivative')
plt.legend()
plt.show()


