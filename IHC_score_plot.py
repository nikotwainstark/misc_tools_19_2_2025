import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

directory_path = r"C:\PythonProjects\PhD Project\Data\IMPORTANT\GLUT\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\ObjectData"
files = os.listdir(directory_path)
files = [f for f in files if f.lower().endswith("csv")]


img_path = r'C:\PythonProjects\PhD Project\Data\IMPORTANT\GLUT\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\ObjectData\ihc_score_stain'
os.makedirs(img_path, exist_ok=True)
scale_factor = 0.1

for file_name in files:

    file_path = os.path.join(directory_path, file_name)
    patient_name = file_path.split('ObjectData\\')[1].split('.vsi')[0]

    chunk_size=20000
    region_data = {}
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk = chunk[['Analysis Region', 'Object Id', 'XMin', 'XMax', 'YMin', 'YMax', 'GLUT1 Cytoplasm OD',
                       'GLUT1 Positive Classification', 'GLUT1 Positive Cytoplasm Classification', 'Classifier Label']]
        # data = data[['Analysis Region','Object Id','XMin','XMax','YMin','YMax',
        #              'GLUT1 Cytoplasm OD', 'GLUT1 Positive Classification',
        #              'GLUT1 Positive Cytoplasm Classification', 'Classifier Label']]
        # chunk['Xabs'] = np.abs(chunk['XMax'] - chunk['XMin'])
        # chunk['Yabs'] = np.abs(chunk['YMax'] - chunk['YMin'])

        for kw in np.unique(chunk['Analysis Region']):
            core = chunk[chunk['Analysis Region'] == kw]
            if kw in region_data:
                region_data[kw] = pd.concat([region_data[kw], core])
            else:
                region_data[kw] = core

    for kw, core in region_data.items():
        output_file = os.path.join(img_path, f'{patient_name}_{kw}.png')
        if os.path.exists(output_file):
            print(f'File {output_file} already exists, skipping region: {kw}')
            continue

        x_max_core = core['XMax'].values.max()
        x_min_core = core['XMin'].values.min()
        y_max_core = core['YMax'].values.max()
        y_min_core = core['YMin'].values.min()

        core_copy = core.copy()
        core_copy['XMax'] = core_copy['XMax'] - x_min_core
        core_copy['YMax'] = core_copy['YMax'] - y_min_core
        core_copy['XMin'] = core_copy['XMin'] - x_min_core
        core_copy['YMin'] = core_copy['YMin'] - y_min_core

        core_copy['XMin'] = (core_copy['XMin'] * scale_factor).astype(int)
        core_copy['XMax'] = (core_copy['XMax'] * scale_factor).astype(int)
        core_copy['YMin'] = (core_copy['YMin'] * scale_factor).astype(int)
        core_copy['YMax'] = (core_copy['YMax'] * scale_factor).astype(int)

        ihc_image = np.zeros((core_copy['XMax'].max(), core_copy['YMax'].max(), 3), dtype=np.uint8)

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
                    ihc_image[xmin:xmax, ymin:ymax] = [0, 255, 255]

        ihc_image_stored = ihc_image / 255.0
        ihc_image_stored = np.flip(ihc_image_stored, axis=1)
        ihc_image_stored = np.rot90(ihc_image_stored, k=1)

        plt.imsave(output_file, ihc_image_stored)
        print(f'Image saved: {output_file}')
    print(f'Done, {patient_name}')



# loading png images to pandas, and export it to Excel sheet.
r1_graded_png_file_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\GLUT\\Halo archive 2024-10-18 12-16 - v4.0.5107 R1 hires\\ObjectData\\ihc_score_stain"
r1_ungraded_png_file_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\GLUT\\Halo archive 2024-10-18 12-16 - v4.0.5107 R1 hires\\ObjectData\\ihc_score_stain_ungraded"
r2_graded_png_file_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\GLUT\\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\\ObjectData\\ihc_score_stain"
r2_ungraded_png_file_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\GLUT\\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\\ObjectData\\ihc_score_stain_ungraded"
png_files = os.listdir("C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\GLUT\\all_pngs_graded")

data_sheet = pd.DataFrame(columns= ["annotation_ref_ungraded_(GLUT_img)_direction", "annotation_ref_(GLUT_img)_direction",
                                    "patient_group", "patient_ID.", "core_No.", "G_status",
                                    "core_locations", "tumour_location", "ir_img_direction", "ir_annotation_img_direction"])
patient_id_lst=[]
ungraded_direction_lst = []
graded_direction_lst = []
patient_group_lst = []
core_no_lst = []
g_status_lst = []
core_locations_lst = []
tumour_locations_lst = []
ir_img_direction_lst = []
annotation_ir_img_direction_lst = []

count = 1

for png_file in png_files:
    s = png_file.split(" ")
    patient_id = s[0]
    patient_id_lst.append(patient_id)
    kw_lst = s[1].split(".")[0].split("_")
    if kw_lst[0] == 'R1':
        graded_direction_lst.append(os.path.join(r1_graded_png_file_path, png_file))
        ungraded_direction_lst.append(os.path.join(r1_ungraded_png_file_path, png_file))
        annotation_ir_img_direction_lst.append(np.nan)  # subject to change
        ir_img_direction_lst.append(np.nan)  # subject to change
    if kw_lst[0] == 'R2':
        graded_direction_lst.append(os.path.join(r2_graded_png_file_path, png_file))
        ungraded_direction_lst.append(os.path.join(r2_ungraded_png_file_path, png_file))
        annotation_ir_img_direction_lst.append(np.nan)  # subject to change, this should point to a .png file
        ir_img_direction_lst.append(np.nan)  # subject to change, this should point to a .dmt file
    core_no_lst.append(count)

    print(kw_lst)
    if len(kw_lst) == 4:
        if kw_lst[-1] == 'Gh':
            g_status_lst.append(1)
        elif kw_lst[-1] == 'Gl':
            g_status_lst.append(0)
        elif kw_lst[-1] == 'Tz':
            g_status_lst.append(3)
        elif kw_lst[-1] == 'met' or kw_lst[-1] == 'met1':
            g_status_lst.append(4)
        elif kw_lst[-1] == 'met2':
            g_status_lst.append(5)
        elif kw_lst[-1] == 'met3':
            g_status_lst.append(6)
        else:
            print("GLUT intensity shows neither Gh , Gl, Tz nor met, check Excel sheet.")
    else:
        g_status_lst.append(np.nan)
    if kw_lst[0] == "R1":
        patient_group_lst.append("R1")
    if kw_lst[0] == "R2":
        patient_group_lst.append("R2")
    if kw_lst[1] == "A":
        tumour_locations_lst.append('primary')
    if kw_lst[1] == "B":
        tumour_locations_lst.append("lymph node")
    core_locations_lst.append(kw_lst[2])
    count += 1

data_sheet["annotation_ref_ungraded_(GLUT_img)_direction"] = ungraded_direction_lst
data_sheet["annotation_ref_(GLUT_img)_direction"] = graded_direction_lst
data_sheet["patient_group"] = patient_group_lst
data_sheet["patient_ID."] = patient_id_lst
data_sheet["core_No."] = core_no_lst
data_sheet["G_status"] = g_status_lst
data_sheet["core_locations"] = core_locations_lst
data_sheet["tumour_location"] = tumour_locations_lst
data_sheet["ir_img_direction"] = ir_img_direction_lst
data_sheet["ir_annotation_img_direction"] = annotation_ir_img_direction_lst

save_path = "C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\file_directions"
os.makedirs(save_path, exist_ok=True)
data_sheet.to_excel(os.path.join(save_path, 'file_directions.xlsx'), index=False)

#  plot images from directory file
test = pd.read_excel("C:\\PythonProjects\\PhD Project\\Data\\IMPORTANT\\file_directions\\file_directions.xlsx")

for row in np.arange(test.shape[0])[:2]:
    ungraded_path = test.iloc[row, 1]
    graded_path = test.iloc[row, 0]
    Image.MAX_IMAGE_PIXELS = None
    img1 = Image.open(ungraded_path)
    img2 = Image.open(graded_path)
    width1, height1 = img1.size
    width2, height2 = img2.size

    new_width = width1 + width2
    new_height = max(height1, height2)
    canvas = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (width1, 0))

    canvas.show()

