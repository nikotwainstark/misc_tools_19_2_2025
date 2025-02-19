import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


path = "C:\PythonProjects\PhD Project\GLUT\Halo archive 2023-06-15 15-12 - v3.5.3577 R1 TMA\ObjectData\C007956 R1_A.vsi_object_Data.csv"
patient_name = path.split('ObjectData\\')[1].split('.vsi')[0]
data = pd.read_csv(path)
data.head()
data = data[['Analysis Region','Object Id','XMin','XMax','YMin','YMax', 'GLUT1 Cytoplasm OD', 'GLUT1 Positive Classification',
                     'GLUT1 Positive Cytoplasm Classification', 'Classifier Label']]
data['Xabs'] = np.abs(data['XMax'] - data['XMin'])
data['Yabs'] = np.abs(data['YMax'] - data['YMin'])
data.head()
data_copy = data.copy()
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

    ihc_image = np.zeros((core_copy['XMax'].max(), core_copy['YMax'].max(), 3))

    for id in core_copy['Object Id']:
        xmin = core_copy.iloc[id]['XMin']
        xmax = core_copy.iloc[id]['XMax']
        ymin = core_copy.iloc[id]['YMin']
        ymax = core_copy.iloc[id]['YMax']
        if core_copy.iloc[id]['Classifier Label'] == 'Epithelium':
            if core_copy.iloc[id]['GLUT1 Positive Classification'] == 0:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 0, 255]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 1:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 255, 0]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 2:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 165, 0]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 3:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 0, 0]
        elif core_copy.iloc[id]['Classifier Label'] == 'Stroma':
            if core_copy.iloc[id]['GLUT1 Positive Classification'] == 0:
                ihc_image[xmin:xmax, ymin:ymax] = [0, 255, 0]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 1:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 255, 255]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 2:
                ihc_image[xmin:xmax, ymin:ymax] = [255, 192, 203]
            elif core_copy.iloc[id]['GLUT1 Positive Classification'] == 3:
                ihc_image[xmin:xmax, ymin:ymax] = [(0,255,255)]

    ihc_image_stored = ihc_image/255
    ihc_image_stored = np.flip(ihc_image_stored, axis=1)
    ihc_image_stored = np.rot90(ihc_image_stored, k=1)
    img_path = 'C:\PythonProjects\PhD Project\GLUT\Halo archive 2023-06-15 15-12 - v3.5.3577 R1 TMA\ObjectData\ihc_score_stain'
    plt.imsave(img_path+f'\{patient_name}_{kw}.png', ihc_image_stored)
print('Done')
