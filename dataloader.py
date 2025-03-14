import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication

sys.path.append(r"C:\PythonProjects\PhD Project\GitHub\new pyir\PyIR\src")
sys.path.append(r"C:\PythonProjects\PhD Project\Haolin's repository\misc_tools_19_2_2025")
from pyir_spectralcollection import PyIR_SpectralCollection
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
        self.tile = None
        self.tissue_mask_lst = []
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

        if int(core_id) is not None:
            core_idx = np.where(self.core_id_arr == core_id)[0]
            print('\n ============The target dataframe=============')
            print(self.data_frame.iloc[core_idx])
            return self.data_frame.iloc[core_idx]
        else:
            return None

    def groupby_data(self, patient_id=None, core_location=None, g_status=None):
        """
        need to change this funciton to a more universal one by allowing user to pass a tuple containing groupby cols
        """
        if not patient_id:
            patient_id = input('Provide patient id (strings: C00 + 4 digits):')
        if not core_location:
            core_location = input('Provide core location (strings: S, Tm, T, B)')
        if not g_status:
            g_status = input(
                'Provide g_status: (numerical values: 0: low, 1: high, 3: T_Tz, 4: met/met1, 5: met2, 6: met3)')

        # Collect the non-None inputs in a dictionary
        groupby_columns = []
        group_values = []

        if patient_id:
            groupby_columns.append('patient_ID.')
            group_values.append(patient_id)
        if core_location:
            groupby_columns.append('core_locations')
            group_values.append(core_location)
        if g_status:
            groupby_columns.append('G_status')
            group_values.append(int(g_status))

        # Check if there are any valid grouping columns and values
        if groupby_columns:
            # Create the grouping and fetch the group
            print(f'\n ============The target dataframe group by {", ".join(groupby_columns)} =============')
            target_df = self.data_frame.groupby(groupby_columns).get_group(tuple(group_values))
            print(target_df)
            return target_df, groupby_columns, group_values
        else:
            print('※※※※※ Invalid input! ※※※※※')  # No valid grouping columns
            return None

    def load_pyir_data(self, path=None, core_id=None):
        """
        if core_id is providing, replace internal attributes with core pyir data

        if a core pyir data path is provided, load from the path

        illegal entry: both path and core_id are supplied.
        """
        if path and not core_id:
            self.tile = PyIR_SpectralCollection(path)
            self.tile.is_loaded()
            print("loaded form path")
        elif core_id and not path:
            self.tile = PyIR_SpectralCollection(self.core_information_finder(core_id).iloc[0, -2])
            self.tile.is_loaded()
            print("loaded by core_id")
        else:
            raise ValueError("should only choose one loading method!")
        self.data = self.tile.data
        self.wavn = self.tile.wavenumbers
        self.xpx = self.tile.xpixels
        self.ypx = self.tile.ypixels
        print("※※※※※ completed ※※※※※")
        return self.tile

    def create_mask(self, lower=1600, upper=1700, data=None, wavn=None, threshold=1.5):
        if not data:
            data = self.data
        if not wavn:
            wavn = self.wavn
        intensity = self.tile.area_between(lower, upper, data, wavn)
        intensity[intensity > 15] = np.mean(intensity)  # replace pixels with area intensity greater than 15 to img average

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        ax[0].imshow(intensity.reshape(self.ypx, self.xpx))
        ax[0].set_title(f'Area intensity between {lower} and {upper}')
        ax[1].imshow((intensity > threshold).reshape(self.ypx, self.xpx))
        ax[1].set_title(f'Area intensity mask between {lower} and {upper} with threshold {threshold}')
        plt.draw()
        plt.pause(0.1)

        QApplication.processEvents()

        return intensity > threshold

    def create_core_mask(self, lower=1600, upper=1700, data=None, wavn=None, threshold=1.5):

    def save_mask(self, mask, path):
        """
        save mask as .png file
        Input:
            mask: boolean 2D array to be saved as png file
            path: system path for saving mask image
        """
        mask_img = mask.reshape(self.ypx, self.xpx).astype(np.uint8) * 255
        cv2.imwrite(path, mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imshow("image", mask_img)
        cv2.waitKey()
        print(f'※※※※※ mask saved at {path} ! ※※※※※')

    @staticmethod
    def load_annotation(path, tissue_mask):
        """
        load annotation from image (.png) path.
        use tissue_mask to filter out incorrect annotation that falls beyond the tissue boundary

        Return: a dictionary contains annotation masks
        """

        annotation_mask_dict = {}
        img_data = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_data = img_data.reshape(-1, 3)

        # hard stroma: (0, 255, 0) = Green
        normal_stroma_mask = np.all(img_data == [0, 255, 0], axis=1)
        annotation_mask_dict['normal stroma'] = normal_stroma_mask * tissue_mask

        # # fibro stroma: (0, 255, 255) = Cyan
        # cancer_stroma_mask = np.all(img_data == [0, 255, 255], axis=1)
        # annotation_mask_dict['cancer stroma'] = cancer_stroma_mask * tissue_mask

        # normal epithelium: (255, 0, 255) = purple
        normal_epithelium_mask = np.all(img_data == [255, 0, 255], axis=1)
        annotation_mask_dict['normal epithelium'] = normal_epithelium_mask * tissue_mask

        # cancer epithelium: (255, 0, 0) = Red
        cancer_epithelium_mask = np.all(img_data == [255, 0, 0], axis=1)
        annotation_mask_dict['cancer epithelium'] = cancer_epithelium_mask * tissue_mask

        # cancer associated stroma: (0, 255, 255) = Cyan
        cancer_stroma_mask = np.all(img_data == [0, 255, 255], axis=1)
        annotation_mask_dict['cancer stroma'] = cancer_stroma_mask * tissue_mask

        # cancer epithelium Glut low: (255, 255, 0) = Yellow
        cancer_epi_glutL_mask = np.all(img_data == [255, 255, 0], axis=1)
        annotation_mask_dict['Cancer epithelium Glut low'] = cancer_epi_glutL_mask * tissue_mask

        # cancer epithelium Glut high: (255, 165, 0) = orange
        cancer_epi_glutH_mask = np.all(img_data == [255, 165, 0], axis=1)
        annotation_mask_dict['cancer epithelium Glut high'] = cancer_epi_glutH_mask * tissue_mask

        img_name = path.split('\\')[-1]
        print(f"loaded {list(annotation_mask_dict.keys())} labels from {img_name}")

        return annotation_mask_dict

    @staticmethod
    def random_sampling_from_mask(mask, sample_size=2000, replace=False):
        """
        randomly sample a specified size of data in user-input mask (default: without replacement)
        Primary goal: take the indexes that match the condition (True) from a given mask (Boolean array)
        and then randomly select a specified number of samples.

        Small data size warning: if the available data is less than 1000, give a warning of overfitting risk.

        Insufficient samples handling: if the required number of samples exceeds the available data,
        determine whether the difference is within the acceptable range, if so, ask whether to oversample,
        otherwise throw an error.

        Normal sampling: when the required number of samples is within the range of available data, random sampling
        is done directly.

        Return: numpy array of randomly sampled indices.
        """
        idx_arr = np.where(mask)[0]
        if idx_arr.shape[0] < 1000:
            print('WARNING: Possible overfitting risk since data size is small than 1000!')

        if sample_size > idx_arr.shape[0]:
            print('Sample size is larger than the mask!')
            print(f'size of annotated pixel in this mask:{idx_arr.shape[0]}')
            if np.abs(sample_size - idx_arr.shape[0]) <= 1000 and idx_arr.shape[0] > 1000:
                user_confirmation = input('Absolute size difference less than 1000 and annotated data larger than 1000,'
                                          ' consider oversample? y/n')
                if user_confirmation == 'y':
                    print('implement oversampling...')
                    return np.random.choice(idx_arr, size=sample_size, replace=True)
                elif user_confirmation == 'n':
                    print('sample all data available in this mask')
                    return idx_arr
            else:
                raise ValueError(
                    f'Required sample size too big, sample size:{sample_size} / mask size: {idx_arr.shape[0]}! '
                    f'consider down sampling or ignore this class')

        return np.random.choice(idx_arr, size=sample_size, replace=replace)

    def make_labels(self, annotation_path, tissue_mask=None, sample_size=3000):
        """
        Make labels for each class in the annotation.

        Defaults to labelling of all mask_dict's keys in the annotation.
        For each key, the user is prompted if random sampling is required:
          - If the user enters 'y', prompt for a sampling value (sample_size by default), the
            If the user enters 'y', prompt for a sample value (sample_size by default) and call self.random_sampling_from_mask;
          - If the user enters 'n', the indexes of all True in the mask are used.

        Returns: annotated_data
          annotated_data: all the sampled data (numpy array) after horizontal splicing.
          labels: array of labels (numpy array).
        """
        annotation_mask_dict = self.load_annotation(annotation_path, tissue_mask)
        label_num = 0
        label_cache = []
        annotated_data_cache = []

        for key in annotation_mask_dict.keys():
            total_count = annotation_mask_dict[key].sum()
            print(f'\nSize of annotated "{key}" data: {total_count}')
            if total_count == 0:
                print(f'"{key}" has empty annotation, pass')
                label_num += 1
                continue

            print(f'↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Label num {label_num} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
            # random sampling hint
            user_choice = input(
                f"Do you want to apply random sampling for '{key}'? data size in {key} is {total_count}. (y/n): ").strip().lower()
            if user_choice == 'y':
                sample_input = input(f"Enter random sampling size for '{key}' (default {sample_size}): ").strip()
                if sample_input == "":
                    sample_size_value = sample_size
                else:
                    try:
                        sample_size_value = int(sample_input)
                    except ValueError:
                        print("Invalid sample size input, using default sample size.")
                        sample_size_value = sample_size
                print(f"Random sampling {sample_size_value} data from '{key}'...")
                random_idx = self.random_sampling_from_mask(annotation_mask_dict[key], sample_size_value)
            else:
                print(f"Using full sampling (all available data) for '{key}'...")
                random_idx = np.where(annotation_mask_dict[key])[0]

            annotated_data_cache.append(random_idx)
            labels = np.full(random_idx.shape[0], label_num)
            label_cache.append(labels)
            label_num += 1

            print('\n※※※※※ Output: annotated INDEX and LABELS ※※※※※')

        if len(annotated_data_cache) > 0:
            annotated_data = np.hstack(annotated_data_cache)
            labels = np.hstack(label_cache)
            return annotated_data, labels
        else:
            print("※※※※※ No data was added to the cache, returning None. ※※※※※")
            return None, None

    def make_grouped_labels(self, patient_id=None, core_location=None, g_status=None, sample_size=1000):
        """
        Grouping the data based on the three values entered by the user (patient_id, core_location, G_status).
        Call make_labels on each group to generate labels and return data and labels for all groups.
        It also returns the group label for each group.
        Regarding sampling: take a user-specified (sample_size) number of each class of each core at a time.
        For example, stroma has 3987 pixels in patient C002419, if the sample_size is specified as
        1000 the programme will randomly sample 1000 cases and iterate this operation for all results after groupby_data filtered data.
        """
        if self.data_frame is None:
            raise ValueError("call load_file_directory first to load the dataframe")

        grouped_df, group_columns, grouped_keys = self.groupby_data(patient_id, core_location, g_status)

        annotated_data_cache = []
        labels_cache = []

        annotation_path_arr = grouped_df['ir_annotation_img_direction'].values
        ir_path_arr = grouped_df['ir_img_direction'].values
        g_status_arr = grouped_df['G_status'].values
        core_location_arr = grouped_df['core_locations'].values
        patient_id_arr = grouped_df['patient_ID.'].values

        for i in range(len(annotation_path_arr)):
            print(f'Proceed to annotation No.{i + 1} from the grouped dataframe {grouped_keys} above.....')
            if pd.isnull(annotation_path_arr[i]) or pd.isnull(ir_path_arr[i]):
                print(f'No annotation/ir image path found in No.{i + 1} annotation!')
                continue
            self.load_pyir_data(ir_path_arr[i])
            print(f'Creating mask for {patient_id_arr[i]}, {core_location_arr[i]}, {g_status_arr[i]} IR image....')
            _ = 1.5
            while True:
                tissue_mask = self.create_mask(threshold=_)
                a = input('Good with this mask? (y/n): ')
                if a == 'y':
                    self.tissue_mask_lst.append(tissue_mask)
                    break
                elif a == 'n':
                    _ = float(input('New threshold: '))

            print(
                f'\n\n============ making label for {patient_id_arr[i]}, {core_location_arr[i]}, {g_status_arr[i]} ============')
            print(f'↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
            annotated_index, labels = self.make_labels(annotation_path_arr[i], tissue_mask=tissue_mask
                                                       , sample_size=sample_size)
            if annotated_index is None or labels is None:
                print(f"Warning: No valid data returned for annotation {annotation_path_arr[i]}")
                continue

            zeros = np.zeros(self.ypx * self.xpx)
            for m in np.unique(labels):  # labels should be integer
                zeros[annotated_index[np.where(labels == m)]] = m + 1
            plt.figure()
            plt.imshow(zeros.reshape(self.ypx, self.xpx))
            plt.title(f'labeling from {patient_id_arr[i]}, {core_location_arr[i]}, {g_status_arr[i]}')
            plt.show(block=False)

            annotated_data = self.data[annotated_index]
            annotated_data_cache.append(annotated_data)
            labels_cache.append(labels)

        print(
            '\n=============MAKE GROUPED LABEL END. Output: core annotated DATA, core LABELS, GROUPED KEYS=============')
        if len(annotated_data_cache) > 0:
            final_data = np.vstack(annotated_data_cache)
            final_label = np.hstack(labels_cache)

            print(f'※※※※※ Grouped by {group_columns}, sample size of each label: {sample_size} ※※※※※\n')

            return final_data, final_label, tuple(grouped_keys)
        else:
            print('※※※※※ No annotated data and labels in the caches, check if annotation exists on images! Returning None. ※※※※※')

    @staticmethod
    def disp_png(path):
        """
        disp low res png. display windows hold still until user presses "Enter" on the keyboard
        """
        image = cv2.imread(path)

        new_width = int(image.shape[1] * 0.05)
        new_height = int(image.shape[0] * 0.05)
        new_size = (new_width, new_height)

        low_res_image = cv2.resize(image, new_size)

        cv2.imshow('Low Resolution Image', low_res_image)
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # The ASCII code of the Enter key is 13.
                break
        cv2.destroyAllWindows()


