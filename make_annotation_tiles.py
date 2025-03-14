# this script is for testing new codes intended for a cleaner pipeline and preprocessing steps
# a master Excel sheet (xlsx) will be needed in order to provide ir dmt file direction ,annotation direction and patient
# metadata. Tools are based on Dr Dougal Ferguson's PyIR toolkit.

import sys

sys.path.append(r"C:\PythonProjects\PhD Project\GitHub\new pyir\PyIR\src")  # replace this address with PyIR root folder
sys.path.append(r"C:\PythonProjects\PhD Project\Haolin's repository\misc_tools_19_2_2025")
import os
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from importlib import reload
reload(smartimage)
reload(dataloader)
reload(spectra_prep)
import time
import csv
import pandas as pd
from matplotlib.widgets import Slider
import pickle
import smartimage
import spectra_prep
import dataloader
from pyir_spectralcollection import PyIR_SpectralCollection
from sklearn.cluster import KMeans


master_xlsx_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\file_directions\\file_directions.xlsx"
# load master Excel sheet
dl = dataloader.Dataloader(master_xlsx_path)
dl.load_file_directory()

image_root_folder_path = "C:\PythonProjects\PhD Project\Data\IMPORTANT\ir_imgs_annotations"

grouped_df, cols, keys = dl.groupby_data()  # should be grouped by patient, then generate a larger tile contains all cores from that patient
grouped_df = grouped_df.fillna(0)
ir_arr = grouped_df['ir_img_direction'].values.astype(str)
g_status_arr = grouped_df['G_status'].values.astype(int)
g_status_arr = g_status_arr.astype(str)
core_location_arr = grouped_df['core_locations'].values
patient_ir_arr = grouped_df['patient_ID.'].values
for i in range(len(ir_arr)):
    if len(ir_arr[i]) <= 5:
        continue

    patient_id = patient_ir_arr[i]
    if core_location_arr[i] == 'S':
        img_name = patient_ir_arr[i] + '_' + core_location_arr[i]
    else:
        img_name = patient_ir_arr[i] + '_' + core_location_arr[i] + '_' + g_status_arr[i]

    tile = dl.load_pyir_data(ir_arr[i])
    si = smartimage.SmartImage(tile)
    full_path_grey = os.path.join(image_root_folder_path, patient_id, img_name + '.png')
    full_path_kmeans = os.path.join(image_root_folder_path, patient_id, img_name + '_kmeans.png')
    area_data = tile.area_between(1600, 1700, tile.data, tile.wavenumbers)
    area_data[area_data >= 15] = np.mean(area_data)

    _ = 1.5
    while True:
        tissue_mask = dl.create_mask(threshold=_)
        a = input('Good with this mask? (y/n): ')
        if a == 'y':
            dl.tissue_mask_lst.append(tissue_mask)
            break
        elif a == 'n':
            _ = float(input('New threshold: '))
        else:
            print('Invalid input')
    pl = spectra_prep.PrepLine(dl.data, dl.wavn, dl.xpx, dl.ypx, tissue_mask)
    pl.apply_mask()
    pl.customised_pipeline(pipeline_dict=func_dict)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pl.data)
    k_labels = kmeans.labels_ + 1
    k_img = pl.rebulid_img_data(k_labels, pl.xpx, pl.ypx, tissue_mask)

    si.export2img(area_data, full_path_grey)
    si.export2img(k_img, full_path_kmeans, if_grey=False)
