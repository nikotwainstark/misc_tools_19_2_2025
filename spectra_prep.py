import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
from sklearn.decomposition import PCA


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
        self.min2zero = False
        self.derivatised = False
        self.normalised = False
        self.denoised = False
        self.baseline_corrected = False
        self.band_cropped = False
        self.pca = None
        self.tissue_mask = tissue_mask

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
            data[row, ] = data - minimum
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
            plt.title("Denoised mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        if use_class_data:
            self.data = denoised_data
            self.denoised = True
            print("Internal data denoised")

        print("PCA denoising completed")

        return (denoised_data, wavn) if not use_class_data else None

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

        zeros = np.zeros(xpx*ypx)

        for i in range(numbers_of_plots):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
            axs[0].plot(wavn, pca_loadings[i,:])
            zeros[tissue_mask == True] = pca_scores[:, i]
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
                lower_idx = np.argmin(np.abs(wavn-np.min(ranges)))
                upper_idx = np.argmin(np.abs(wavn-np.max(ranges)))
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
            plt.title("Band keeping mean spectrum")
            plt.xlabel("Wavenumber(cm-1)")
            plt.ylabel("Absorbance")
            plt.show()

        return (cropped_data, cropped_wavn) if not use_class_data else None

    def band_cropping(self, band_list=[900, 1350, "", 1490, 1800], data=None, wavn=None, visual=False):
        """
        band cropping given user input list. "" in list means cropping the region.
        example: [1200, 1350, "", 1490, 1800]
        """
        use_class_data = data is None and wavn is None
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
                cropped_data, cropped_wavn = self.band_removing([lower, upper], cropped_data, cropped_wavn, visual=False)

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


    # def polyfit_baseline_correction(self,
    #                                 data=None,
    #                                 wavn=None,
    #                                 polyorder=3,
    #                                 max_iter=100,
    #                                 tol=0.01,
    #                                 visual=False):
    #     """
    #     simple polynomial fitting for baseline estimation and correction
    #     """
    #     print("initialising polyfit baseline correction")
    #
    #     use_class_data = (data is None and wavn is None)
    #     if data is None:
    #         data = self.data
    #     if wavn is None:
    #         wavn = self.wavn
    #
    #     baseline = data.copy()
    #
    #     for i in range(max_iter):
    #         coefs = np.polynomial.polynomial.polyfit(wavn, baseline.T, deg=polyorder)
    #         fitted_curve = np.polynomial.polynomial.polyval(wavn, coefs)
    #
    #         mask = baseline > fitted_curve
    #         new_baseline = np.where(mask, fitted_curve, baseline)
    #
    #         p = np.linalg.norm(new_baseline - baseline) / np.linalg.norm(baseline)
    #         print(f"Iteration {i+1}, Convergence p = {p}")
    #
    #         if p < tol:
    #             break
    #
    #         baseline = new_baseline
    #
    #     corrected_data = data - baseline
    #
    #     if use_class_data:
    #         self.data = corrected_data
    #         self.baseline_corrected = True
    #
    #     if visual:
    #         plt.figure(figsize=(8, 5))
    #         plt.plot(wavn, np.mean(data, axis=0), label="Original (mean)")
    #         plt.plot(wavn, np.mean(baseline, axis=0), label="Estimated Baseline (mean)")
    #         plt.plot(wavn, np.mean(corrected_data, axis=0), label="Corrected (mean)")
    #         plt.title(f"Baseline-corrected spectrum (poly order={polyorder}, tol={tol})")
    #         plt.xlabel("Wavenumber (cm⁻¹)")
    #         plt.ylabel("Absorbance")
    #         plt.legend()
    #         plt.show()
    #
    #     return corrected_data, baseline

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

                # 显示当前收敛值
                pbar.set_postfix({"Convergence": f"{p:.4f}"})

                if p < tol:
                    # 如果提前收敛，则将进度条直接更新到 100%
                    pbar.n = pbar.total
                    pbar.refresh()
                    break

                baseline = new_baseline
                pbar.update(1)  # 正常迭代进度

            # 如果循环是正常走完的（没有 break），在这里也把进度条更新到 100%
            if pbar.n < pbar.total:
                pbar.n = pbar.total
                pbar.refresh()

            # 可选：暂停片刻，确保用户能看到 100% 状态（仅为视觉效果）
            time.sleep(0.2)

        corrected_data = data - baseline

        if use_class_data:
            self.data = corrected_data
            self.baseline_corrected = True

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

        return corrected_data, baseline

    def customised_pipeline(self, pipeline_dict, should_return=True):
        """
        根据用户给定的字典执行预处理流程，
        字典的键为函数名称，值为该函数所需的参数字典。

        例如：
        pipeline_dict = {
            "smooth": {"window": 7},
            "normalize": {"method": "z-score"},
            "baseline_correction": {"polyorder": 4}
        }
        """
        print("Initialising preprocessing pipeline....\n")
        count=1

        for func_name, params in pipeline_dict.items():
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                if callable(func):
                    print(f"Step {count}: Executing {func_name} with parameters '{params}' ... \n")
                    if params is None:
                        func()
                    else:
                        func(**params)
                else:
                    print(f"{func_name} not a callable object！")
            else:
                print(f"Warning: {func_name} 在 spectra_prep.PrepLine 中不存在！")
            print("\n----------------------------------------------------------\n")
            count += 1
        print("Preprocess sequence completed")
        if should_return:
            return self.data, self.wavn

    def fast_pipeline(self, should_return=True):
        func_dict = {"pca_denoising": {"visual": True},
                     "polyfit_baseline_correction": {"max_iter":10, "visual": True},
                     "band_cropping": {"visual": True},
                     "normalisation": {"visual": True},
                     "sg_deriv": {"visual":True},
                     "band_removing": {"ranges": [1330, 1520], "visual": True}}
        self.customised_pipeline(func_dict, should_return)

    def plot_mean_spectrum(self, data=None, wavn=None):
        use_class_data = (data is None and wavn is None)
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





    # reload(spectra_prep)test = spectra_prep.PrepLine(tile.data, tile.wavenumbers, tile.xpixels, tile.ypixels, tissue_mask=tissue_mask)
    #


    # def mnf_denoising
    # def emsc_baseline_correction
    # def mie_extinction_correction
    # def spectra_make_annotation
    # def default_spectra_prepline









