import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import scipy
from sklearn.decomposition import PCA
import pywt
import numpy as np
import asym_pls
from smartimage import SmartImage
import air_pls
import h5py
import sys

sys.path.append("C:\\PythonProjects\\PhD Project\\GitHub\\new pyir\PyIR\src")
from pyir_spectralcollection import PyIR_SpectralCollection
from importlib import reload

reload(asym_pls)
reload(air_pls)


class PrepLine:
    def __init__(self, data, wavn, xpx, ypx, tissue_mask):

        if not isinstance(wavn, np.ndarray) or wavn.ndim != 1:
            raise ValueError("wavn must be a 1D numpy array")

        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array")

        if data is not None and len(data.shape) == 3:
            print("\ndata must be a 2D numpy array, reshaping data now...\n")
            data = data.reshape(-1, len(wavn))

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")

        if data.shape[1] != wavn.shape[0]:
            raise ValueError(f"Mismatch: data.shape[1]={data.shape[1]}, wavn.shape[0]={wavn.shape[0]}")

        for _ in tqdm(range(6), desc="loading", ncols=60):
            time.sleep(0.1)
        print("input data check completed, object initialised")

        self.data = data
        self.wavn = wavn
        self.xpx = xpx
        self.ypx = ypx
        self.masked = False
        self.min2zero = False
        self.derivatised = False
        self.normalised = False
        self.denoised = False
        self.baseline_corrected = False
        self.band_lst = [900, 1350, "", 1490, 1800]
        self.band_cropped = False
        self.pca = None
        self.tissue_mask = tissue_mask
        self.func_dict = {
            'polyfit_baseline_correction': {'max_iter': 10, 'tol': 0.001, 'visual': True},
            'band_cropping': {'band_list': [1000, 1340, '', 1490, 1800], 'visual': True},
            "normalisation": {"visual": True},
            "pca_denoising": {"visual": True}
            # "sg_deriv": {"visual":True},
            # "band_removing": {"ranges": [1330, 1520], "visual": True}
        }

    def apply_mask(self, data=None, mask=None):
        use_class_data = data is None and mask is None
        data = self.data if data is None else data
        mask = self.tissue_mask if mask is None else mask

        data_masked = data[mask]

        if use_class_data:
            self.data = data_masked
            self.masked = True

        print("data masked")
        return data_masked if not use_class_data else None

    def point_min2zero(self, data=None, wavn=None):

        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        min_test = np.min(data)
        data = data - min_test

        if use_class_data:
            self.min2zero = True
            self.data = data

        print("minimised dataset down to zero (point wise)")
        return data, wavn if not use_class_data else None

    def spectral_min2zero(self, data=None, wavn=None):
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        for row in range(data.shape[0]):
            minimum = np.min(data[row, :])
            data[row,] = data - minimum
        if use_class_data:
            self.min2zero = True
            self.data = data
        print("minimised dataset down to zero (spectral wise)")
        return data, wavn if not use_class_data else None

    def sg_deriv(self, data=None, wavn=None, n_deriv=2, window_size=21, polyorder=5, visual=False):
        """
        Iteratively call numpy gradient function for nth derivatives

        """
        print("Initialising savitzky-golay(smoothed) derivatisation")

        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        deriv_data = scipy.signal.savgol_filter(data, window_size, polyorder, n_deriv, axis=1)

        if window_size > data.shape[1]:
            raise ValueError(f"window_size ({window_size}) cannot be larger than data.shape[1] ({data.shape[1]})")

        crop_point = (window_size - 1) // 2
        deriv_data = deriv_data[:, crop_point:-crop_point]

        deriv_wavn = wavn[crop_point:-crop_point]

        if use_class_data:
            self.data = deriv_data
            self.derivatised = True
            self.wavn = deriv_wavn
            print("Internal data derivatised")

        if visual:
            plt.figure()
            plt.plot(deriv_wavn, np.mean(deriv_data, axis=0))
            plt.title(f"Derivatised (order:{n_deriv}) mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        print(f"SG derivative order:{n_deriv}, completed.")

        return (deriv_data, deriv_wavn) if not use_class_data else None

    def gradient_deriv(self, data=None, wavn=None, n_deriv=2, visual=False):
        """
        Perform gradient derivatisation
        """

        print("Initialising gradient((smoothed)) derivatisation")

        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        for _ in range(n_deriv):
            data = np.gradient(data, wavn, axis=1)

        if visual:
            plt.figure()
            plt.plot(wavn, np.mean(data, axis=0))
            plt.title(f"Derivatised (order:{n_deriv}) mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        if use_class_data:
            self.data = data
            self.derivatised = True
            print("derivtive data stored")

        print(f"gradient derivative order:{n_deriv}, completed.")

        return (data, wavn) if not use_class_data else None

    def pca_core(self, n_components=50):
        """
        Define the pca object for other functions
        """
        self.pca = PCA(n_components)
        print("PCA core initialised")

    def pca_denoising(self, data=None, wavn=None, visual=False):
        """
        Perform pca denoising to the data, number of components defined by self.pca_core
        """
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        if not self.pca:
            self.pca_core()
            print("pca core auto-initialised, number of components by default: 50")

        pca = self.pca
        transformed_data = pca.fit_transform(data)
        denoised_data = pca.inverse_transform(transformed_data)

        if visual:
            plt.figure()
            plt.plot(wavn, np.mean(denoised_data, axis=0))
            plt.title("Denoised(PCA) mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        if use_class_data:
            self.data = denoised_data
            self.denoised = True
            print("Internal data denoised")

        print("PCA denoising completed")

        return (denoised_data, wavn) if not use_class_data else None

    def wt_denoising(self, data=None, wavn=None, wavelet='sym5', level=5, threshold_scale=1, visual=False):
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        if level is None:
            level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)

        coeffs = pywt.wavedec(data, wavelet, level=level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data))) * threshold_scale

        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            new_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))

        denoised_spectrum = pywt.waverec(new_coeffs, wavelet)

        if visual:
            plt.figure()
            plt.plot(wavn, np.mean(denoised_spectrum[:len(data)], axis=0))
            plt.title("Denoised(wavelet transform) mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        if use_class_data:
            self.data = denoised_spectrum[:len(data)]
            self.denoised = True
            print("Internal data denoised")

        return (denoised_spectrum[:len(data)], wavn) if not use_class_data else None

    def fast_mnf_denoising(self, hyperspectraldata=None, wavn=None, SNR=5, bands=0, visual=False):
        """
        from Dougal's code:
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
            If bands =/= 0, the number of components is set to bands.
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
        # >>> hyperspectraldata = np.random.rand(100, 100, 50)  # A 3D hyperspectral image
        # >>> clean_data = fast_mnf_denoise(hyperspectraldata)
        # >>> clean_data.shape
        (100, 100, 50)
        """

        use_class_data = hyperspectraldata is None
        hyperspectraldata = self.data if hyperspectraldata is None else hyperspectraldata
        wavn = self.wavn if wavn is None else wavn

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
        if bands != 0:
            K = bands
        else:
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

        if visual:
            plt.figure()
            plt.plot(wavn, np.mean(clean_data, axis=0))
            plt.title("Denoised(MNF) mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        if use_class_data:
            self.data = clean_data
            self.denoised = True
            print("Internal data denoised")

        print("PCA denoising completed")
        return (clean_data, wavn) if not use_class_data else None

    def pca_plots(self, data=None, wavn=None, xpx=None, ypx=None, tissue_mask=None, num_plots=10, exp_var=True):
        """
        Perform pca scores and loadings plot for the first N components.
        """
        internal_data = data is None
        internal_wavn = wavn is None
        internal_xpx = xpx is None
        internal_ypx = ypx is None
        internal_mask = tissue_mask is None

        if internal_data:
            data = self.data
        if internal_wavn:
            wavn = self.wavn
        if internal_xpx:
            assert hasattr(self, "xpx"), "Error: xpx is not set in the class!"
            xpx = self.xpx
        if internal_ypx:
            assert hasattr(self, "ypx"), "Error: ypx is not set in the class!"
            ypx = self.ypx
        if internal_mask:
            tissue_mask = self.tissue_mask

        if not self.pca:
            self.pca_core()
            print("pca core auto-initialised, number of components by default: 50")

        pca = self.pca
        pca.fit(data)
        pca_loadings = pca.components_
        pca_scores = pca.transform(data)
        pca_ex_var = pca.explained_variance_ratio_
        pca_cumsum_var = np.cumsum(pca_ex_var)

        numbers_of_plots = min(num_plots, pca.components_.shape[0])

        zeros = np.zeros(xpx * ypx)

        for i in range(numbers_of_plots):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
            axs[0].plot(wavn, pca_loadings[i, :])
            zeros[tissue_mask] = pca_scores[:, i]
            axs[1].imshow(zeros.reshape(ypx, xpx), cmap="coolwarm")
            axs[0].set_title(f"scores for No.{i} PC")
            axs[1].set_title(f"loadings for No.{i} PC. Var: {pca_ex_var[i]:.2%}")
            axs[0].set_xlabel("Wavenumber(cm-1)")
            axs[0].set_ylabel("A.U")
        plt.tight_layout()
        plt.show()

        if exp_var:
            plt.figure()
            plt.plot(pca_cumsum_var, marker="o")
            plt.title("Explained Variance Ratio")
            plt.xlabel("number of components")
            plt.ylabel("Cumulative variance explained")
            plt.show()

        print("Pca plots completed")

    def band_keeping(self, ranges, data=None, wavn=None, visual=False):
        """
        keeps the band ranges from lower to upper limit
        """
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        if data.shape[1] != wavn.shape[0]:
            raise ValueError(f"Mismatch: data.shape[1]={data.shape[1]}, wavn.shape[0]={wavn.shape[0]}")

        if isinstance(ranges, list):
            if len(ranges) == 2:
                lower_idx = np.argmin(np.abs(wavn - np.min(ranges)))
                upper_idx = np.argmin(np.abs(wavn - np.max(ranges)))
                cropped_wavn = wavn[lower_idx:upper_idx]
                cropped_data = data[:, lower_idx:upper_idx]
            else:
                raise ValueError(f"the length of list should be 2 (now: {len(ranges)}), i.e., upper and lower limits")
        else:
            raise TypeError(f"range input should be list (now: {type(ranges)})")

        if use_class_data:
            self.data = cropped_data
            self.wavn = cropped_wavn
            self.band_cropped = True
            print("Internal data and wavenumbers cropped")

        if visual:
            plt.figure()
            plt.plot(cropped_wavn, np.mean(cropped_data, axis=0))
            plt.title("Band keeping mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        return (cropped_data, cropped_wavn) if not use_class_data else None

    def band_removing(self, ranges, data=None, wavn=None, visual=False):
        """
        removes the band ranges from lower to upper limit
        """
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        if data.shape[1] != wavn.shape[0]:
            raise ValueError(f"Mismatch: data.shape[1]={data.shape[1]}, wavn.shape[0]={wavn.shape[0]}")

        if isinstance(ranges, list) and len(ranges) == 2:
            lower_idx = np.argmin(np.abs(wavn - np.min(ranges)))
            upper_idx = np.argmin(np.abs(wavn - np.max(ranges)))

            cropped_wavn = np.concatenate((wavn[:lower_idx], wavn[upper_idx:]))
            cropped_data = np.delete(data, np.s_[lower_idx:upper_idx], axis=1)
        else:
            raise ValueError(f"Invalid range: {ranges}. It must be a list of two values.")

        if use_class_data:
            self.data = cropped_data
            self.wavn = cropped_wavn
            self.band_cropped = True
            print("Internal data and wavenumbers cropped")

        if visual:
            plt.figure()
            plt.plot(cropped_wavn, np.mean(cropped_data, axis=0))
            plt.title("keeping mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        return (cropped_data, cropped_wavn) if not use_class_data else None

    def band_cropping(self, band_list=None, data=None, wavn=None, visual=False):
        """
        band cropping given user input list. "" in list means cropping the region.
        example: [1200, 1350, "", 1490, 1800]
        """
        use_class_data = data is None and wavn is None and band_list is None
        band_list = self.band_lst
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        if len(band_list) < 2:
            raise ValueError("band_list needs at least two integer values")

        min_band = np.min([b for b in band_list if b != ""])
        max_band = np.max([b for b in band_list if b != ""])

        print(f"Keeping range: {min_band}-{max_band}")
        cropped_data, cropped_wavn = self.band_keeping([min_band, max_band], data, wavn, visual=False)

        for i in range(1, len(band_list) - 1):
            if band_list[i] == "":
                lower = band_list[i - 1]
                upper = band_list[i + 1]

                print(f"Removing range: {lower}-{upper}")
                cropped_data, cropped_wavn = self.band_removing([lower, upper], cropped_data, cropped_wavn,
                                                                visual=False)

        if use_class_data:
            self.data = cropped_data
            self.wavn = cropped_wavn
            self.band_cropped = True
            print("Internal data and wavenumbers cropped")

        if visual:
            plt.figure()
            plt.plot(cropped_wavn, np.mean(cropped_data, axis=0))
            plt.title(f"Cropped Spectrum. Range: {band_list}")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.show()

        print("Band cropping completed")
        return (cropped_data, cropped_wavn) if not use_class_data else None

    def normalisation(self, data=None, wavn=None, norm="vector", f_index=1655, visual=False):
        """
        Feature normalisation or vector normalisation. (Default: vector normalisation)
        - If `data` and `wavn` are provided, function returns normalised data.
        - If `data` and `wavn` are None, function modifies `self.data`.
        """
        use_class_data = data is None and wavn is None
        data = self.data if data is None else data
        wavn = self.wavn if wavn is None else wavn

        norms, ref_values = None, None

        if norm == "vector":
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalised_data = data / norms

            if use_class_data:
                self.data = normalised_data
                self.normalised = True
                print(f"Internal data vector normalised.")

        elif norm == "feature":
            idx = np.argmin(np.abs(wavn - f_index))
            if not (0 <= idx < data.shape[1]):
                raise ValueError(f"{idx} out of index range, data matrix only has {data.shape[1]} columns")
            ref_values = data[:, idx].reshape(-1, 1)
            ref_values[ref_values == 0] = 1
            normalised_data = data / ref_values

            if use_class_data:
                self.data = normalised_data
                self.normalised = True
                print(f"Internal data feature normalised.")

        else:
            raise ValueError("Currently only support 'vector' and 'feature' normalisation.")

        if visual:
            plt.figure(figsize=(8, 5))
            if norm == "vector" and norms is not None:
                plt.plot(wavn, np.mean(normalised_data, axis=0))
            elif norm == "feature" and ref_values is not None:
                plt.plot(wavn, np.mean(normalised_data, axis=0))
            plt.title(f"{norm.capitalize()} Normalised Mean Spectrum")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.show()

        return normalised_data if not use_class_data else None

    def polyfit_baseline_correction(self,
                                    data=None,
                                    wavn=None,
                                    polyorder=3,
                                    max_iter=100,
                                    tol=0.01,
                                    visual=False):
        """
        simple polynomial fitting for baseline estimation and correction
        """
        print("initialising polyfit baseline correction")

        use_class_data = (data is None and wavn is None)
        if data is None:
            data = self.data
        if wavn is None:
            wavn = self.wavn

        baseline = data.copy()

        with tqdm(total=max_iter, desc="Baseline correction", unit="iter") as pbar:
            for i in range(max_iter):
                coefs = np.polynomial.polynomial.polyfit(wavn, baseline.T, deg=polyorder)
                fitted_curve = np.polynomial.polynomial.polyval(wavn, coefs)

                mask = baseline > fitted_curve
                new_baseline = np.where(mask, fitted_curve, baseline)

                p = np.linalg.norm(new_baseline - baseline) / np.linalg.norm(baseline)

                pbar.set_postfix({"Convergence": f"{p:.4f}"})

                if p < tol:
                    pbar.n = pbar.total
                    pbar.refresh()
                    break

                baseline = new_baseline
                pbar.update(1)

            if pbar.n < pbar.total:
                pbar.n = pbar.total
                pbar.refresh()

        corrected_data = data - baseline
        if visual:
            plt.figure(figsize=(8, 5))
            plt.plot(wavn, np.mean(data, axis=0), label="Original (mean)")
            plt.plot(wavn, np.mean(baseline, axis=0), label="Estimated Baseline (mean)")
            plt.plot(wavn, np.mean(corrected_data, axis=0), label="Corrected (mean)")
            plt.title(f"Baseline-corrected spectrum (poly order={polyorder}, tol={tol})")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.legend()
            plt.show()

        if use_class_data:
            self.data = corrected_data
            self.baseline_corrected = True

        print('complete polynomial fitting baseline correction!')

        return corrected_data, baseline if not use_class_data else None

    def asym_pls_correction(self, data=None,
                            wavn=None,
                            lam=1e6,
                            max_iter=10,
                            p=0.001,
                            tol=1e-6,
                            d=2,
                            n_jobs=-1,
                            visual=False
                            ):

        """
        implement baseline correction with paralleled asymmetric partial least square baseline estimation
        """
        use_class_data = (data is None and wavn is None)
        if data is None:
            data = self.data
        if wavn is None:
            wavn = self.wavn

        estimator = asym_pls.AsymmetricPlsEstimator(lam=lam, p=p, d=d, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        baseline = estimator.asysm_parallel(data.T)
        corrected_data = data - baseline.T

        if visual:
            plt.figure(figsize=(8, 5))
            plt.plot(wavn, np.mean(data, axis=0), label="Original (mean)")
            plt.plot(wavn, np.mean(baseline.T, axis=0), label="Estimated Baseline (mean)")
            plt.plot(wavn, np.mean(corrected_data, axis=0), label="Corrected (mean)")
            plt.title(f"Asymmetric PLS Baseline-corrected spectrum (maxiter={max_iter}, tol={tol})")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.legend()
            plt.show()

        if use_class_data:
            self.data = corrected_data
            self.baseline_corrected = True

        print("complete Asymmetric PLS baseline correction!")
        return (corrected_data, baseline) if not use_class_data else None

    def airpls_correction(self, data=None,
                          wavn=None,
                          lam=200,
                          max_iter=300,
                          porder=1,
                          n_jobs=-1,
                          visual=False):
        use_class_data = (data is None and wavn is None)
        if data is None:
            data = self.data
        if wavn is None:
            wavn = self.wavn

        estimator = air_pls.AirPLSEstimator(data.T, lam, porder, max_iter, n_jobs)
        baseline = estimator.airPLS_parallel()
        corrected_data = data - baseline.T
        print(corrected_data.shape)

        if visual:
            plt.figure(figsize=(8, 5))
            plt.plot(wavn, np.mean(data, axis=0), label="Original (mean)")
            plt.plot(wavn, np.mean(baseline.T, axis=0), label="Estimated Baseline (mean)")
            plt.plot(wavn, np.mean(corrected_data, axis=0), label="Corrected (mean)")
            plt.title(f"AirPLS Baseline-corrected spectrum (maxiter={max_iter},lam={lam}, p order={porder})")
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Absorbance")
            plt.legend()
            plt.show()

        if use_class_data:
            self.data = corrected_data
            self.baseline_corrected = True

        print("complete AirPLS baseline correction!")
        return (corrected_data, baseline) if not use_class_data else None

    def customised_pipeline(self, pipeline_dict, should_return=True):
        for func_name, params in pipeline_dict.items():
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                if callable(func):
                    print(f"Executing {func_name} with parameters: {params}")
                    if params:
                        func(**params)
                    else:
                        func()
                else:
                    print(f"{func_name} not callable")
            else:
                print(f"Warning: {func_name} not in PrepLine!")
        return self.data, self.wavn if should_return else None

    def fast_pipeline(self, should_return=True):

        self.customised_pipeline(self.func_dict, should_return)

    @staticmethod
    def print_pipeline_example():
        func_dict = {
            'polyfit_baseline_correction': {'max_iter': 10, 'tol': 0.001, 'visual': True},
            'band_cropping': {'band_list': [1000, 1340, '', 1490, 1800], 'visual': True},
            "normalisation": {"visual": True},
            "pca_denoising": {"visual": True},
            "sg_deriv": {"visual": True},
            "band_removing": {"ranges": [1330, 1520], "visual": True}
        }
        print(func_dict)

    def plot_mean_spectrum(self, data=None, wavn=None):
        if data is None:
            data = self.data
        if wavn is None:
            wavn = self.wavn

        plt.figure()
        plt.plot(wavn, np.mean(data, axis=0))
        plt.title("Mean spectrum of data")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Absorbance")
        plt.show()

    def rebulid_img_data(self, data=None, xpx=None, ypx=None, tissue_mask=None, visual=True):
        """
        reshape 2d spectrum to a 3D image
        """

        if data is None:
            data = self.data
        if xpx is None:
            xpx = self.xpx
        if ypx is None:
            ypx = self.ypx
        if tissue_mask is None:
            tissue_mask = self.tissue_mask

        zero = np.zeros(ypx * xpx)
        zero[tissue_mask] = data
        img = zero.reshape((ypx, xpx))

        if visual:
            plt.figure()
            plt.imshow(img)
            plt.show()

        return img

    def rebuild_spectra_data(self, tissue_data=None, wavn=None, xpx=None, ypx=None, tissue_mask=None, clickon=True):
        if tissue_data is None:
            tissue_data = self.data
        if xpx is None:
            xpx = self.xpx
        if ypx is None:
            ypx = self.ypx
        if tissue_mask is None:
            tissue_mask = self.tissue_mask

        zero = np.zeros((xpx * ypx, wavn.shape[0]))
        zero[tissue_mask] = tissue_data

        if clickon:
            tile = PyIR_SpectralCollection()
            tile.data = zero
            tile.wavenumbers = wavn
            tile.xpixels = xpx
            tile.ypixels = ypx
            si = SmartImage(tile)
            si.clickon()

        return zero

    def save2hdf5(self, data=None, wavn=None):
        use_class_data = (data is None and wavn is None)
        if data is None:
            data = self.data
        if wavn is None:
            wavn = self.wavn

    # reload(spectra_prep)test = spectra_prep.PrepLine(tile.data, tile.wavenumbers, tile.xpixels, tile.ypixels, tissue_mask=tissue_mask)
    # def emsc_baseline_correction
    # def mie_extinction_correction
    # def spectra_make_annotation
    # def default_spectra_prepline
