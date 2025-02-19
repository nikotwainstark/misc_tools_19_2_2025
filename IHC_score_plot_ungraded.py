import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

counts_overall = {}
folder = "C:\PythonProjects\PhD Project\GLUT\Halo archive 2024-10-18 12-16 - v4.0.5107 R1 hires\ObjectData"
for file_name in os.listdir(folder):
    if file_name.endswith('.csv'):
        path = os.path.join(folder +'\\'+ file_name)
        patient_name = path.split('ObjectData\\')[1].split('.vsi')[0]
        data = pd.read_csv(path)
        data.head()
        data = data[['Analysis Region','Object Id','XMin','XMax','YMin','YMax', 'GLUT1 Cytoplasm OD', 'GLUT1 Positive Classification',
                             'GLUT1 Positive Cytoplasm Classification', 'Classifier Label']]
        data['Xabs'] = np.abs(data['XMax'] - data['XMin'])
        data['Yabs'] = np.abs(data['YMax'] - data['YMin'])
        data.head()

        # #check histogram
        # counts, bins = np.histogram(data_copy['GLUT1 Cytoplasm OD'])
        # counts_overall[file_name[:6]] = counts
        # bin_width = bins[1] - bins[0]
        # plt.figure()
        # plt.bar(bins[:-1], counts, width=bin_width, edgecolor='black', alpha=0.7)
        # plt.show()

        # #check histogram counts per patient
        # counts_overall

        # od_value_max = data['GLUT1 Cytoplasm OD'].values.max()
        # od_value_min = data['GLUT1 Cytoplasm OD'].values.min()

        data['GLUT1 Cytoplasm OD'] = (data['GLUT1 Cytoplasm OD'] - data['GLUT1 Cytoplasm OD'].min())/ (data['GLUT1 Cytoplasm OD'].max() - data['GLUT1 Cytoplasm OD'].min())
        data_copy = data.copy()
        # kw for indicating glut high/low and constituent regions
        keywords = np.unique(data_copy['Analysis Region'])
        for kw in keywords:
            core = data_copy[data['Analysis Region'] == kw]
            x_max_core = core['XMax'].values.max()
            x_min_core = core['XMin'].values.min()
            y_max_core = core['YMax'].values.max()
            y_min_core = core['YMin'].values.min()

            core_copy = core.copy()
            core_copy['XMax'] = core_copy['XMax'] - x_min_core
            core_copy['YMax'] = core_copy['YMax'] - y_min_core
            core_copy['XMin'] = core_copy['XMin'] - x_min_core
            core_copy['YMin'] = core_copy['YMin'] - y_min_core
            ihc_image = np.zeros((core_copy['XMax'].max(), core_copy['YMax'].max(), 4))

            for id in core_copy['Object Id']:
                xmin = core_copy.iloc[id]['XMin']
                xmax = core_copy.iloc[id]['XMax']
                ymin = core_copy.iloc[id]['YMin']
                ymax = core_copy.iloc[id]['YMax']
                alpha_value = core_copy.iloc[id]['GLUT1 Cytoplasm OD']
                if core_copy.iloc[id]['Classifier Label'] == 'Epithelium':
                        ihc_image[xmin:xmax, ymin:ymax] = [255, 0, 255, alpha_value *255 ]
                elif core_copy.iloc[id]['Classifier Label'] == 'Stroma':
                        ihc_image[xmin:xmax, ymin:ymax] = [(0, 255, 0, alpha_value *255 )]

            ihc_image_stored = ihc_image/255

            ihc_image_stored = np.flip(ihc_image_stored, axis=1)
            ihc_image_stored = np.rot90(ihc_image_stored, k=1)
            img_path = 'C:\PythonProjects\PhD Project\GLUT\Halo archive 2024-10-18 12-16 - v4.0.5107 R1 hires\ObjectData\ihc_score_stain_ungraded'
            ihc_image_stored = np.ascontiguousarray(ihc_image_stored)
            plt.imsave(img_path+f'\{patient_name}_{kw}.png', ihc_image_stored)
        print(f'Done, {patient_name}')









