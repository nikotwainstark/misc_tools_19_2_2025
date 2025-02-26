import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

counts_overall = {}

img_path = r'C:\PythonProjects\PhD Project\Data\IMPORTANT\GLUT\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\ObjectData\ihc_score_stain_ungraded'
os.makedirs(img_path, exist_ok=True)

folder = r"C:\PythonProjects\PhD Project\Data\IMPORTANT\GLUT\Halo archive 2024-10-18 12-37 - v4.0.5107 R2 hires\ObjectData"

scale_factor = 0.1

for file_name in os.listdir(folder):

    if file_name.endswith('.csv'):
        path = os.path.join(folder, file_name)
        patient_name = path.split('ObjectData\\')[1].split('.vsi')[0]

        # 逐块读取 CSV 文件，而不是一次性加载整个文件
        region_data = {}  # 用来累积每个区域的数据

        chunk_size=20000
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            chunk = chunk[['Analysis Region', 'Object Id', 'XMin', 'XMax', 'YMin', 'YMax', 'GLUT1 Cytoplasm OD',
                           'GLUT1 Positive Classification', 'GLUT1 Positive Cytoplasm Classification', 'Classifier Label']]
            chunk.loc[:, 'GLUT1 Cytoplasm OD'] = (chunk['GLUT1 Cytoplasm OD'] - chunk['GLUT1 Cytoplasm OD'].min()) / \
                                                  (chunk['GLUT1 Cytoplasm OD'].max() - chunk['GLUT1 Cytoplasm OD'].min())
            for kw in np.unique(chunk['Analysis Region']):
                core = chunk[chunk['Analysis Region'] == kw]
                if kw in region_data:
                    region_data[kw] = pd.concat([region_data[kw], core])
                else:
                    region_data[kw] = core
        # print(region_data)
        # 遍历所有累积的区域数据生成图像
        for kw, core in region_data.items():
            output_file = os.path.join(img_path, f'{patient_name}_{kw}.png')
            if os.path.exists(output_file):
                print(f'File {output_file} already exists, skipping region: {kw}')
                continue

            x_min_core = core['XMin'].min()
            y_min_core = core['YMin'].min()

            core_copy = core.copy()
            core_copy['XMax'] = core_copy['XMax'] - x_min_core
            core_copy['YMax'] = core_copy['YMax'] - y_min_core
            core_copy['XMin'] = core_copy['XMin'] - x_min_core
            core_copy['YMin'] = core_copy['YMin'] - y_min_core

            core_copy['XMin'] = (core_copy['XMin'] * scale_factor).astype(int)
            core_copy['XMax'] = (core_copy['XMax'] * scale_factor).astype(int)
            core_copy['YMin'] = (core_copy['YMin'] * scale_factor).astype(int)
            core_copy['YMax'] = (core_copy['YMax'] * scale_factor).astype(int)

            # 重新计算图像尺寸
            img_width = core_copy['XMax'].max()  # 或者可以加上1，确保包含所有像素
            img_height = core_copy['YMax'].max()
            ihc_image = np.zeros((img_width, img_height, 4), dtype=np.uint8)

            # 根据缩放后的坐标填充图像
            for _, row in core_copy.iterrows():
                xmin, xmax, ymin, ymax = row['XMin'], row['XMax'], row['YMin'], row['YMax']
                alpha_value = row['GLUT1 Cytoplasm OD']
                if row['Classifier Label'] == 'Epithelium':
                    ihc_image[xmin:xmax, ymin:ymax] = [255, 0, 255, int(alpha_value * 255)]
                elif row['Classifier Label'] == 'Stroma':
                    ihc_image[xmin:xmax, ymin:ymax] = [0, 255, 0, int(alpha_value * 255)]

            # 图像保存前可进行旋转、翻转等处理
            ihc_image_stored = np.flip(np.rot90(ihc_image / 255.0, k=1), axis=1)
            ihc_image_stored = np.ascontiguousarray(ihc_image_stored)

            # 保存图像
            plt.imsave(output_file, ihc_image_stored)
            print(f'Image saved: {output_file}')
        print(f'Done, {patient_name}')








