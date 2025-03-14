import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import ipywidgets as widgets
from IPython.display import display
from pyir_spectralcollection import PyIR_SpectralCollection
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import ipywidgets as widgets
from IPython.display import display
import cv2
import os


class SmartImage:
    def __init__(self, tile):
        self.tile = tile
        self.wavenumbers = tile.wavenumbers
        self.spectra_collection = pd.DataFrame(columns=self.wavenumbers)
        self.current_spectrum = None

    def sub_cap(self):
        """
        This function captures sub square of IR image. Returns a new PyIR_Spectralcollection Obj
        so that you don't need to worry about the original data.

        :return: ndims, tile_sub
        """
        self.tile.update_sums()
        self.tile.disp_image(self.tile.totalimage)
        points = plt.ginput(2)
        fstp = np.round(points)[0].astype(int)
        secp = np.round(points)[1].astype(int)
        data = self.tile.data
        if data.ndim == 2:
            data = self.tile.reshaper_3D(
                data, [self.tile.ypixels, self.tile.xpixels, -1]
            )
        sub_data = data[fstp[1]: secp[1], fstp[0]: secp[0], :]
        ndims, sub_2d_data = self.tile.reshaper_3D(sub_data)

        tile_sub = PyIR_SpectralCollection()
        tile_sub.data = sub_2d_data
        tile_sub.wavenumbers = self.tile.wavenumbers
        tile_sub.xpixels = ndims[1]
        tile_sub.ypixels = ndims[0]

        return ndims, tile_sub

    def clickon(self):
        """
        clickon function inspired by Alex's code.
        Left click to display the spectrum at the pixel. Right click to store selected spectrum in the spectra collection
        :return: return spectral collection: Pandas DataFrame.
        """
        data = self.tile.data
        ypixels = self.tile.ypixels
        xpixels = self.tile.xpixels

        if data.ndim == 2:
            feature_data = data.reshape(ypixels, xpixels, -1)
        elif data.ndim == 3:
            feature_data = data
        else:
            raise ValueError("Input data must be a 2D or 3D array.")

        image_data = feature_data.sum(axis=2)

        output = widgets.Output()

        fig, (ax_image, ax_curve) = plt.subplots(2, 1, figsize=(8, 12))
        plt.subplots_adjust(bottom=0.2)
        ax_image.imshow(image_data, cmap="grey"
                        , extent=[0, xpixels, ypixels, 0]
                        )

        (line,) = ax_curve.plot([], [])
        ax_curve.set_title("Spectrum")
        ax_curve.set_xlabel("wavenumbers")
        ax_curve.set_ylabel("Absorbance")

        saved_spectra_text = ax_curve.text(
            0.05, 0.95, "", transform=ax_curve.transAxes, verticalalignment="top"
        )

        @output.capture()
        def onclick(event):
            if event.inaxes == ax_image:
                x, y = int(event.xdata), int(event.ydata)
                if event.button == 1:  # left click
                    feature_curve = feature_data[y, x, :]
                    update_feature_curve(x, y, feature_curve)
                    # print(f"Left click at ({x}, {y})")  # Debug information
                elif event.button == 3:  # right click
                    if self.current_spectrum is not None:
                        x, y, spectrum = self.current_spectrum
                        index = y * xpixels + x  # 2d array to 1d array
                        self.spectra_collection.loc[index] = spectrum
                        print(f"Spectrum at ({x}, {y}) saved with index {index}.")
                        update_saved_spectra_text()
                    else:
                        print("No spectrum to save.")
                    # print(f"Right click at ({x}, {y})")  # Debug information
            # click on spectrum curve and update the image with regard to the wavenumber (x value)

        def update_feature_curve(x, y, feature_curve):
            line.set_data(self.wavenumbers, feature_curve)
            ax_curve.relim()
            ax_curve.autoscale_view()
            ax_curve.set_title(f"Spectrum at ({x}, {y})")
            fig.canvas.draw()
            self.current_spectrum = (x, y, feature_curve)
            # print(f"Spectrum updated at ({x}, {y}).")  # Debug information

        def update_saved_spectra_text():
            saved_spectra_text.set_text(
                f"Saved spectra: {len(self.spectra_collection)}"
            )
            fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        display(output)
        plt.show()

        return self.spectra_collection

    def flip_spectra(self, replace=False):
        """
        Filp the spectra order of plots. Note that it does not flip the wavenumbers attribute.
        :param replace: replace the flipped data within the attribute spectra_collection
        :return: Plot the spectrum. No returns.
        """

        spectra = self.spectra_collection.values
        waven = self.spectra_collection.columns.values
        index = self.spectra_collection.index.values

        spectra = np.flip(spectra, axis=1)
        waven = np.flip(waven)

        new_df = pd.DataFrame(spectra, index=index, columns=waven)
        if replace:
            self.spectra_collection = new_df
        return new_df

    def show_spectra_collection(
            self, data=None, highlight=False, mean=False, flip=False, label=False
    ):
        """
        Show spectra plot of the spectra data (Default: attribute: spectra_collection) with highlighted wavenumbers.
        :param data: Optional. The data to plot. Can be a pandas DataFrame, Series, or numpy array.
        :param highlight: list. The peaks need to be highlighted.
        :param mean: bool. Plot the mean spectra if True.
        :param flip: if True, return reversed (with respect to X-axis) plot.
        :param label: if True, add labels to each spectra.
        :return: draw a spectra collection plot.
        """
        if data is None:
            spectra = self.spectra_collection.values
            waven = self.spectra_collection.columns.values
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                print("The collection is empty.")
                return
            spectra = data.values
            waven = data.columns.values
        elif isinstance(data, pd.Series):
            if data.empty:
                print("The collection is empty.")
                return
            spectra = data.values.reshape(1, -1)
            waven = data.columns.values
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                print("The collection is empty.")
                return
            spectra = data
            waven = self.wavenumbers
        else:
            print("Data needs to be pandas DataFrame/Series or Numpy array")
            return

        if not mean and spectra.shape[0] >= 10000:
            user_input = input("\nNumber of Spectrum >= 10000, continue? y/n: ")
            if user_input.lower() != "y":
                return

        if mean:
            mean_spectra = np.mean(spectra, axis=0)
            plt.plot(waven, mean_spectra, label="Spectra mean")
            plt.title("Mean of Spectra Collection")
        else:
            for i in range(spectra.shape[0]):
                number = str(i)
                filler = "th"
                if number.endswith("1") and not number.endswith("11"):
                    filler = "st"
                elif number.endswith("2") and not number.endswith("12"):
                    filler = "nd"
                elif number.endswith("3") and not number.endswith("13"):
                    filler = "rd"
                plt.plot(waven, spectra[i, :], label=f"the {i}{filler} spectrum")

            plt.title("Spectra Collection")

        plt.xlabel("Wavenumber/cm-1")
        plt.ylabel("Absorbance")

        if highlight and isinstance(highlight, list):
            for wavenumber in highlight:
                plt.axvline(x=wavenumber, color="r", linestyle="--")
                plt.text(
                    wavenumber + 2,
                    plt.ylim()[0],
                    s=str(wavenumber),
                    color="r",
                    ha="center",
                    va="bottom",
                    rotation=45,
                )

        if flip:
            plt.gca().invert_xaxis()
        if label:
            plt.legend()
        plt.show()

    def export2mat(self, file_path, data=None, wavenumbers=None):
        """
        :param file_path: file path that stores the .mat file
        :param data: any data to be converted into .mat file. Default: self.spectra_collection
        :param wavenumbers: wavenumbers data for .mat file. If data is DataFrame, it does not need this parameter.
        :return: save a file to designated location. No return values.
        """
        if data is None:
            data = self.spectra_collection

        if isinstance(data, pd.DataFrame):
            if data.empty:
                print("The collection is empty.")
                return
            data_values = data.values
            wavn = data.columns.astype(float).values
            data_idx = data.index.values.astype(str)
            mat_name = str(input(".mat MATLAB variable's name:"))

            # Create a dictionary with all required data
            mdict = {
                mat_name: data_values,
                "wavenumbers": wavn,
                "idx_of_data_by_row": data_idx,
            }

        elif isinstance(data, np.ndarray):
            # Create a dictionary for numpy array data, index will not be specified.
            if data.size == 0:
                print("The collection is empty.")
                return
            if wavenumbers is None:
                print("Please input wavenumbers")
                return
            mat_name = str(input(".mat dictionary name:"))
            mdict = {
                mat_name: data,
                "wavenumbers": wavenumbers,
            }

        else:
            print(
                "\nSpectra collection data needs to be in Numpy array or Pandas DataFrame"
            )
            return

        # Save to .mat file
        savemat(file_path, mdict)
        print(f"\nData successfully saved to {file_path}")
        return mdict

    # def importmat(self, path):

    def release_spectral_collection(self):
        self.spectra_collection = pd.DataFrame(columns=self.wavenumbers)
        if self.spectra_collection.empty:
            print("\nspectral collection data is wiped out.")

    def load_all_spectra(self):
        """
        Load and replace the current data with the full data of the tile
        """

        if self.spectra_collection.empty:
            self.spectra_collection = pd.DataFrame(
                self.tile.data,
                columns=self.wavenumbers,
                index=np.arange(self.tile.data.shape[0]),
            )
            print("\nSpectra replaced")
        else:
            _ = input(
                "Note this will replace all of your selected spectra with total spectra of the image, continue? y/n"
            )
            if _.lower() == "y":
                self.spectra_collection = pd.DataFrame(
                    self.tile.data,
                    columns=self.wavenumbers,
                    index=np.arange(self.tile.data.shape[0]),
                )
                print("\nSpectra replaced")
                del _
            elif _.lower() == "n":
                del _

    # def settick:

    def create_idx_img(self, idx, idx_data, y=0, x=0, full_size=0, return_tile=False):
        """
        Define a simple filter that generates the image corresponding to the given indexed data. Support hyperspectral image.
        if return_tile is True, return the hyperspectral tile
        """
        if x == 0 & y == 0:
            y = self.tile.ypixels
            x = self.tile.xpixels
        if full_size == 0:
            full_size = self.tile.data.shape[0]

        if idx_data.ndim == 1:
            empty = np.zeros(full_size)
            empty[idx] = idx_data
            empty = empty.reshape(y, x)
            plt.figure()
            plt.imshow(empty)
            plt.show()

        elif idx_data.ndim == 2:
            empty = np.zeros(full_size, idx_data.shape[-1])
            empty[idx] = idx_data
            empty = np.sum(empty, axis=1)
            empty = empty.reshape(y, x)
            plt.figure()
            plt.imshow(empty)
            plt.show()
            if return_tile:
                new_tile = PyIR_SpectralCollection()
                new_tile.data = empty
                new_tile.ypixels = y
                new_tile.xpixels = x
                new_tile.wavenumbers = self.wavenumbers
                return new_tile

    def export2img(self, data, save_path, if_grey=True):
        """
        Export a numpy array (1-D) to a proper sized image.
        If if_grey is True, return a grey scale image.
        If if_grey is False, return a color image (e.g., for kmeans labels).
        """
        min_val = np.min(data)
        max_val = np.max(data)

        # Normalization
        normalised_data = (data - min_val) / (max_val - min_val)

        # Check if we should return grey scale or color image
        if if_grey:
            # Create grey scale image
            grey_data = (normalised_data * 255).astype(np.uint8)

            # Ensure the directory exists
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the grey scale image
            success = cv2.imwrite(save_path, grey_data.reshape(self.tile.ypixels, self.tile.xpixels))

            if success:
                print(f'Grey image saved at {save_path}')
            else:
                print(f'Grey image not saved')

        else:
            # Create a color image using a colormap
            color_data = cv2.applyColorMap((normalised_data * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Ensure the directory exists
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the color image
            success = cv2.imwrite(save_path, color_data.reshape(self.tile.ypixels, self.tile.xpixels, 3))

            if success:
                print(f'Color image saved at {save_path}')
            else:
                print(f'Color image not saved')


class MaskEditor:

    def __init__(self, mask, y, x):
        self.original_mask = mask.copy()
        self.mask = mask.copy().reshape((y, x))
        self.history = [self.mask.copy()]
        self.fig, self.ax = plt.subplots()
        self.clicks = []
        self.confirmation_count = 0

        self.ax.imshow(self.get_display_image(), cmap="gray")

        # Buttons
        self.ax_undo = plt.axes((0.6, 0.05, 0.1, 0.075))
        self.ax_done = plt.axes((0.71, 0.05, 0.1, 0.075))
        self.btn_undo = Button(self.ax_undo, "<")
        self.btn_done = Button(self.ax_done, "√")
        self.btn_undo.on_clicked(self.undo)
        self.btn_done.on_clicked(self.done)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def get_display_image(self):
        display_image = np.zeros((*self.mask.shape, 3))
        display_image[self.mask == True] = [1, 1, 1]  # White where mask is True
        display_image[self.mask == False] = [0, 0, 0]  # Black where mask is False
        return display_image

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        self.clicks.append((x, y))

        if len(self.clicks) == 2:
            self.redraw()

    def update_mask(self):
        if len(self.clicks) == 2:
            (x1, y1), (x2, y2) = self.clicks
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            self.mask[ymin: ymax + 1, xmin: xmax + 1] = False
            self.history.append(self.mask.copy())
            self.clicks = []

    def redraw(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            self.ax.clear()
            display_image = self.get_display_image()

            if len(self.clicks) == 2:
                (x1, y1), (x2, y2) = self.clicks
                xmin, xmax = sorted([x1, x2])
                ymin, ymax = sorted([y1, y2])
                # Draw rectangle in red
                display_image[ymin: ymax + 1, xmin] = [1, 0, 0]
                display_image[ymin: ymax + 1, xmax] = [1, 0, 0]
                display_image[ymin, xmin: xmax + 1] = [1, 0, 0]
                display_image[ymax, xmin: xmax + 1] = [1, 0, 0]

            for x, y in self.clicks:
                display_image[y, x] = [1, 0, 0]  # Mark clicks in red

            self.ax.imshow(display_image, cmap="gray")
            plt.draw()

    def undo(self, event):
        if len(self.history) > 1:
            self.history.pop()
            self.mask = self.history[-1].copy()

            if self.fig is not None and plt.fignum_exists(self.fig.number):
                self.ax.clear()
                self.ax.imshow(self.get_display_image(), cmap="gray")
                self.clicks = []
                plt.draw()

    def done(self, event):
        if len(self.clicks) == 2:
            self.update_mask()
            self.redraw()
            self.confirmation_count = 0  # Reset confirmation count after updating mask
        else:
            self.confirmation_count += 1
            if self.confirmation_count >= 2:
                plt.close(self.fig)

    def start_editing(self):
        plt.show()

    def get_edited_mask(self):
        return self.mask.flatten()

    # def tissue_mask_creator(self):

    # def area_between(self):
