import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\PythonProjects\PhD Project\GitHub\new pyir\PyIR\src")
sys.path.append(r"C:\PythonProjects\PhD Project\Haolin's repository\misc_tools_19_2_2025")
from pyir_spectralcollection import PyIR_SpectralCollection
from matplotlib.widgets import Button
import cv2

class Dataloader:
    """
    dataloader for loading file directory, pyir data, isolating cores and stitching cores
    """
    def __init__(self, file_directory_path):
        self.data = None
        self.wavn = None
        self.xpx = None
        self.ypx = None
        self.annotation_img_paths = None
        self.ir_img_paths = None
        self.core_id_arr = None
        self.ungraded_img_paths = None
        self.patient_id_arr = None
        self.graded_img_paths = None
        self.file_directory_path = file_directory_path
        self.data_frame = None
        self.tissue_mask = None
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

        if a core pyir data path is provided, load from the path

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
        self.tissue_mask = tile.area_between(1600, 1700, tile.data, tile.wavenumbers) > 1.5
        self.data = tile.data
        self.wavn = tile.wavenumbers
        self.xpx = tile.xpixels
        self.ypx = tile.ypixels
        print("completed")
        return tile


    def load_annotation(self, path, tissue_mask=None):
        """
        load annotation from image (.png) path.

        Return: a dictionary contains annotation masks
        """
        if not tissue_mask:
            mask = self.tissue_mask
        else:
            mask = tissue_mask


        annotation_mask_dict = {}

        img_data = cv2.imread(path, cv2.IMREAD_UNCHANGED).reshape(-1, 4)[:, :-1]
        # normal epithelium: (255, 0, 255) = purple
        normal_epithelium_mask = np.all(img_data == [255, 0, 255], axis=1)
        annotation_mask_dict['normal epithelium'] = normal_epithelium_mask * mask

        # normal stroma: (0, 255, 0) = Green
        normal_stroma_mask = np.all(img_data == [0, 255, 0], axis=1)
        annotation_mask_dict['normal stroma'] = normal_stroma_mask * mask

        # cancer epithelium: (255, 0, 0) = Red
        cancer_epithelium_mask = np.all(img_data == [255, 0, 0], axis=1)
        annotation_mask_dict['cancer epithelium'] = cancer_epithelium_mask * mask

        # cancer stroma: (255,165,0) = Orange
        cancer_stroma_mask = np.all(img_data == [255, 165, 0], axis=1)
        annotation_mask_dict['cancer stroma'] = cancer_stroma_mask * mask

        return annotation_mask_dict

    @staticmethod
    def random_sampling_from_mask(mask, sample_size=5000, replace=False):
        """
        randomly sample a specified size of data in user-input mask (default: without replacement)

        Return: numpy array of randomly sampled indices.
        """
        idx_arr = np.where(mask)[0]
        if idx_arr.shape[0] < 1000:
            print('Possible overfitting risk since data size is small than 1000!')

        if sample_size > idx_arr.shape[0]:
            print('Sample size is larger than the mask!')
            print(f'size of annotated pixel in this mask:{idx_arr.shape[0]}')
            if np.abs(sample_size - idx_arr.shape[0]) <= 1000 and idx_arr.shape[0] > 1000:
                user_confirmation = input('absolute size difference less than 1000 and annotated data larger than 1000,'
                                          ' consider oversample? y/n')
                if user_confirmation == 'y':
                    print('implement oversampling...')
                    return np.random.choice(idx_arr, size=sample_size, replace=True)
                elif user_confirmation == 'n':
                    print('sample all data available in this mask')
                    return idx_arr
            else:
                raise ValueError(f'required sample size too big {sample_size} - {idx_arr.shape[0]}! '
                                 f'consider down sampling or ignore this class')

        return np.random.choice(idx_arr, size=sample_size, replace=replace)

    def make_labels(self, annotation_path, sample_size=1000):
        """
        make labels with for each class within the dictionary
        Input: a 1-D or 2-D numpy array, row as

        """
        annotation_mask_dict = self.load_annotation(annotation_path)
        label_num=0
        label_cache = []
        annotated_data_cache = []
        for key in annotation_mask_dict.keys():
            print(f'size of annotated {key} data: {annotation_mask_dict[key].sum()}')
            if annotation_mask_dict[key].sum() == 0:
                print(f'{key} has empty annotation, pass')
                continue
            print(f'random sampling {sample_size} data from {key}...')
            random_idx = self.random_sampling_from_mask(annotation_mask_dict[key], sample_size)
            random_data = self.data[random_idx]
            annotated_data_cache.append(random_data)

            labels = np.full(random_data.shape[0], label_num)
            label_cache.append(labels)
            label_num += 1
        annotated_data = np.vstack(annotated_data_cache)
        labels = np.hstack(label_cache)

        return annotated_data, labels

















