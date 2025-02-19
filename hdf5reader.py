import sys

import os

header = str(os.getcwd())

sub_module = header + r'\Documents\GitHub'

full_path = sub_module + r'\PyIR\src'

sys.path.append(full_path)

module_path = r"C:\GitHub\new pyir\PyIR\src"
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append(r"C:\PythonProjects\PhD Project\GitHub_test")

import pyir_spectralcollection as pir

import pyir_image as pir_im

import pyir_mask as pir_mask

import pyir_pca as pir_pca

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import pickle


train_directory = pd.read_excel(r'Z:\Group Members\Dougal_Ferguson\Prostate Project\annotations\annotation_directory.xlsx')

dougals_hdf5_fold = r'Z:\Group Members\Dougal_Ferguson\Prostate Project\HDF5 Data'
dougals_label_fold = r'Z:\Group Members\Dougal_Ferguson\Prostate Project\annotations'

test_read = pir.PyIR_SpectralCollection(dougals_hdf5_fold+'\Slide '+ str(train_directory['Slide'][0]) + '\DataCube\c' +
                                        str(train_directory['Core Id'][0]) + '_p' + str(train_directory['Patient ID'][0]) +
                                        str(train_directory['Ass'][0]))
test_read.load_hdf5()

test_read.data, test_read.wavenumbers = test_read.keep_range(3800,950, test_read.data, test_read.wavenumbers)


# normal epithelium is green so [0, 255, 0, 255]

# normal stroma is purple so [128, 0, 128, 255]

# cancerous epithelium is pink so [255, 0, 255, 255]

# cancer associated stroma is blue so [0, 0, 255, 255]

# blood is red so [255, 0, 0, 255]

# corpora amylacea is orange so [255, 165, 0, 255]

# crushed tissue is cyan so [0, 255, 255, 255]

# immune infiltrate is yellow so [255, 255, 0, 255]

# necrotic debris with immune infiltration is dark red so [127.5, 0, 0, 255]

### epiths are now refined due to the mislabelling issue

#(see epithelial_only_refining_regression.py)

stroma_n_even = np.empty((1, test_read.wavenumbers.shape[0]))

stroma_c_even = np.empty((1, test_read.wavenumbers.shape[0]))

epith_n_refined = np.empty((1, test_read.wavenumbers.shape[0]))

epith_c3_refined = np.empty((1, test_read.wavenumbers.shape[0]))

epith_c4_refined = np.empty((1, test_read.wavenumbers.shape[0]))

epith_c5_refined = np.empty((1, test_read.wavenumbers.shape[0]))

immune_infiltrate = np.empty((1, test_read.wavenumbers.shape[0]))

nec_debris = np.empty((1, test_read.wavenumbers.shape[0]))

crushed_tissue = np.empty((1, test_read.wavenumbers.shape[0]))

blood = np.empty((1, test_read.wavenumbers.shape[0]))

amylacea = np.empty((1, test_read.wavenumbers.shape[0]))

number_to_sample = 500

for i in np.arange(0, train_directory.shape[0]):

    print(str(i+1) + " of " + str(train_directory.shape[0]))

    test_read = pir.PyIR_SpectralCollection(dougals_hdf5_fold+'\Slide '+ str(train_directory['Slide'][i]) + '\DataCube\c' +
                                        str(train_directory['Core Id'][i]) + '_p' + str(train_directory['Patient ID'][i]) +
                                        str(train_directory['Ass'][i]))

    test_read.load_hdf5()

    test_read.data, test_read.wavenumbers = test_read.keep_range(3800,950, test_read.data, test_read.wavenumbers)

    labels = np.asarray(Image.open(dougals_label_fold+'\Slide '+ str(train_directory['Slide'][i]) + '\core ' +
                                        str(train_directory['Core Id'][i]) + ' annotations.png')).reshape(

        test_read.ypixels*test_read.xpixels, 4)

    # COLOUR ARRAYS ARE [R, G, B, A]

    # normal epithelium is green so [0, 255, 0, 255]

    epith_n_filter =  np.sum((labels== [0, 255, 0, 255]), axis=1)==4

    # normal stroma is purple so [128, 0, 128, 255]

    stroma_n_filter =  np.sum((labels== [128, 0, 128, 255]), axis=1)==4

    # cancerous epithelium is pink so [255, 0, 255, 255]

    epith_c_filter =  np.sum((labels== [255, 0, 255, 255]), axis=1)==4

    # cancer associated stroma is blue so [0, 0, 255, 255]

    stroma_c_filter =  np.sum((labels== [0, 0, 255, 255]), axis=1)==4

    # blood is red so [255, 0, 0, 255]

    blood_filter =  np.sum((labels== [255, 0, 0, 255]), axis=1)==4

    # corpora amylacea is orange so [255, 165, 0, 255]

    amylacea_filter =  np.sum((labels== [255, 165, 0, 255]), axis=1)==4

    # crushed tissue is cyan so [0, 255, 255, 255]

    crushed_tissue_filter =  np.sum((labels== [0, 255, 255, 255]), axis=1)==4

    # immune infiltrate is yellow so [255, 255, 0, 255]

    immune_infiltrate_filter = np.sum((labels== [255, 255, 0, 255]), axis=1)==4

    # necrotic debris with immune infiltration is dark red so [127, 0, 0, 255]

    nec_debris_immune_inf_filter = np.sum((labels== [127, 0, 0, 255]), axis=1)==4


    epith_n_refined =  np.append(epith_n_refined,

        test_read.apply_mask(test_read.data,

                            epith_n_filter[test_read.tissue_mask]), axis = 0)

    stroma_n_even =  np.append(stroma_n_even,

        test_read.apply_mask(test_read.data,

                             stroma_n_filter[test_read.tissue_mask]), axis = 0)

    if train_directory['primary_path'][i] == '3':

        epith_c3_refined =  np.append(epith_c3_refined,

            test_read.apply_mask(test_read.data,

                                 epith_c_filter[test_read.tissue_mask]), axis = 0)

    if train_directory['primary_path'][i] == '4':

         epith_c4_refined =  np.append(epith_c4_refined,

             test_read.apply_mask(test_read.data,

                                  epith_c_filter[test_read.tissue_mask]), axis = 0)

    if train_directory['primary_path'][i] == '5':

        epith_c5_refined =  np.append(epith_c5_refined,

            test_read.apply_mask(test_read.data,

                                 epith_c_filter[test_read.tissue_mask]), axis = 0)

    stroma_c_even =  np.append(stroma_c_even,

       test_read.apply_mask(test_read.data,

                             stroma_c_filter[test_read.tissue_mask]), axis = 0)


    immune_infiltrate =  np.append(immune_infiltrate,

        test_read.apply_mask(test_read.data,

                             immune_infiltrate_filter[test_read.tissue_mask]), axis = 0)

    nec_debris =  np.append(nec_debris,

        test_read.apply_mask(test_read.data,

                             nec_debris_immune_inf_filter[test_read.tissue_mask]), axis = 0)

    crushed_tissue =  np.append(crushed_tissue,

        test_read.apply_mask(test_read.data,

                             crushed_tissue_filter[test_read.tissue_mask]), axis = 0)

    blood =  np.append(blood,

        test_read.apply_mask(test_read.data,

                             blood_filter[test_read.tissue_mask]), axis = 0)

    amylacea =  np.append(amylacea,

        test_read.apply_mask(test_read.data,

                             amylacea_filter[test_read.tissue_mask]), axis = 0)

stroma_n_even = stroma_n_even[1:,:]

stroma_c_even = stroma_n_even[1:,:]

epith_n_refined = epith_n_refined[1:,:]

epith_c3_refined = epith_c3_refined[1:,:]

epith_c4_refined = epith_c4_refined[1:,:]

epith_c5_refined = epith_c5_refined[1:,:]

immune_infiltrate = immune_infiltrate[1:,:]

nec_debris = nec_debris[1:,:]

crushed_tissue = crushed_tissue[1:,:]

blood = blood[1:,:]

amylacea = amylacea[1:,:]

