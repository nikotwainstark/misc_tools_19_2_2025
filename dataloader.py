import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\PythonProjects\PhD Project\GitHub\new pyir\PyIR\src")
sys.path.append(r"C:\PythonProjects\PhD Project\Haolin's repository\misc_tools_19_2_2025")
from pyir_spectralcollection import PyIR_SpectralCollection
from matplotlib.widgets import Button


class Dataloader:
    """
    dataloader for loading file directory, pyir data, isolating cores and stitching cores
    """
    def __init__(self, file_directory_path):
        self.data = None
        self.wavn = None
        self.xpx = None
        self.ypx = None
        self.tissue_mask = None
        self.annotation_img_paths = None
        self.ir_img_paths = None
        self.core_id_arr = None
        self.ungraded_img_paths = None
        self.patient_id_arr = None
        self.graded_img_paths = None
        self.file_directory_path = file_directory_path
        self.data_frame = None
        if not self.data_frame:
            print("please initialised data with load_file_directory")

    def load_file_directory(self, file_directory_path=None):
        """
        load data from class internal directory path. Alternatively user may pass other directory file in the same format
        """
        if not file_directory_path:
            print("loading internal file directory path to internal attributes......")
            file_directory = pd.read_excel(self.file_directory_path)
            self.data_frame = file_directory
        if file_directory_path:
            print("loading external file directory path to internal attributes......")
            self.data_frame = pd.read_excel(file_directory_path)

        self.ungraded_img_paths = self.data_frame.iloc[:, 0]
        self.graded_img_paths = self.data_frame.iloc[:, 1]
        self.annotation_img_paths = self.data_frame.iloc[:, -1]
        self.ir_img_paths = self.data_frame.iloc[:, -2]
        self.core_id_arr = self.data_frame['core_No.']
        self.patient_id_arr = self.data_frame['patient_ID.']

        print("\nCompleted!")

    def core_information_finder(self, core_id=None):
        """
        show core information by provided core id.
        """
        if self.data_frame.empty:
            raise ValueError("No internal data loader, please use load_file_directory!")

        if core_id is not None:
            core_idx = np.where(self.core_id_arr == core_id)[0]
            return self.data_frame.iloc[core_idx]
        else:
            return None

    def load_pyir_data(self, path=None, core_id=None):
        """
        if core_id is providing, replace internal attributes with core pyir data

        if a core pyir data path is provided, load from the pass

        illegal entry: both path and core_id are supplied.
        """
        if path and not core_id:
            tile = PyIR_SpectralCollection(path)
            tile.is_loaded()
            print("loaded form path")
        elif core_id and not path:
            tile = PyIR_SpectralCollection(self.core_information_finder(core_id).iloc[0, -2])
            tile.is_loaded()
            print("loaded by core_id")
        else:
            raise ValueError("should only choose one loading method!")

        self.data = tile.data
        self.wavn = tile.wavenumbers
        self.xpx = tile.xpixels
        self.ypx = tile.ypixels
        print("completed")
        return tile

    def random_sampling_by_class(self):

















