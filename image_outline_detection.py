import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import pandas as pd
from matplotlib.widgets import Slider

module_path = r"C:\GitHub\new pyir\PyIR\src"
if module_path not in sys.path:
    sys.path.append(module_path)
sub_path = "C:\PythonProjects\PhD Project\Haolin's repository\misc_tools_19_2_2025"
sys.path.append(sub_path)
import smartimage
from pyir_spectralcollection import PyIR_SpectralCollection
path = "Z:\\Group Members\\Anna Zetterstrom [t75307az]\\Colorectal\\Colorectal\\R1C4_to_R2C6\\r1c4_to_r2c6.dmt"
tile = PyIR_SpectralCollection(path)
tile.is_loaded()

tissue_mask = tile.area_between(1600,1700, tile.data, tile.wavenumbers) >= 1

tissue_data = tile.data[tissue_mask]
tissue_data = tile.min2zero(tissue_data)
tissue_data, wavn = tile.remove_wax(tissue_data, tile.wavenumbers)
tissue_data, wavn = tile.keep_range(900,1800, tissue_data, tile.wavenumbers)
tissue_data = tile.vector_norm(tissue_data)

mean_centered_data = tissue_data - np.mean(tissue_data, axis=0)
image_rebulid = np.zeros_like(tile.data[:,:tissue_data.shape[1]])
image_rebulid[tissue_mask, :] = tissue_data


test_difference_column = image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[:-1,:,:] - image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[1:,:,:]
plt.figure()
plt.imshow(np.abs(test_difference_column[:,:,325].reshape(tile.ypixels-1,tile.xpixels))>0.05)
plt.show()
test_difference_row = image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[:,:-1,:] - image_rebulid.reshape(tile.ypixels, tile.xpixels,-1)[:,1:,:]
plt.figure()
plt.imshow(np.abs(test_difference_row[:,:,325].reshape(tile.ypixels,tile.xpixels-1))>0.05)
plt.show()


img = tissue_mask.reshape(tile.ypixels, tile.xpixels)

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.imsave('C:\\tissue_outline test\\image.png', img)

image_path = 'C:\\tissue_outline test\\image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (3, 3), 0)

edges = cv2.Canny(blurred, 20, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# plt.figure()
# plt.imshow(edges.reshape(tile.ypixels, tile.xpixels))
# plt.show()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours,  key=lambda x: cv2.arcLength(x, True), reverse=True)
# if len(contours) > 1:
#     # The second largest contour is at index 1
#     max_contour = contours[0]
#     second_largest_contour = contours[1]
#     third_largest_contour = contours[2]
#     fourth_largest_contour = contours[3]
#     # Create a blank image to draw the second largest contour
# else:
#     print("Not enough contours found!")

output = np.zeros_like(image)
cv2.drawContours(output,
                 [contours[0]]
                 + [contours[1]]
                 + [contours[2]]
                 + [contours[3]]
                 + [contours[4]]
                 + [contours[5]]
                 + [contours[6]]
                 + [contours[7]]
                 , -1, 1
                 , thickness=cv2.FILLED
                 )
closed_output = cv2.morphologyEx(output, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

# plt.figure(figsize=(6, 6))
# plt.imshow(closed_output, cmap='gray')
# plt.title("Improved Outer Circular Edge")
# plt.axis('off')
# plt.title(path[-6:-4])
# plt.show()

contours2, _2 = cv2.findContours(closed_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
selected_contours = []
for c in contours2:
    # length = cv2.arcLength(c, True)
    length = len(c)
    if length > 100:
        selected_contours.append(c)
flattened_selected_contours = np.array([item for sublist in selected_contours for item in sublist])
if contours2:
    # Find the longest contour
    # longest_contour = max(contours2, key=lambda x: cv2.arcLength(x, True))
    longest_contour = np.array(flattened_selected_contours)
    # Create a blank image (mask) to draw the contour
    mask = np.zeros_like(closed_output, dtype=np.uint8)
    # Draw the longest contour on the mask
    cv2.drawContours(mask, [longest_contour], -1, (1)
                     # , thickness=cv2.FILLED
                     )
    # Define the structuring element for closing operation (a kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the size of the kernel if necessary
    # Apply the morphological closing operation to the mask
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_contours = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    closed_contour = closed_contours[0][0]
    mask = np.zeros_like(closed_output, dtype=np.uint8)
    cv2.drawContours(mask, [closed_contour], -1, (1)
                     , thickness=cv2.FILLED
                     )
    # Display the result
    # plt.figure(figsize=(6, 6))
    # plt.imshow(mask, cmap='gray')
    # plt.title("Tissue Segment")
    # plt.axis('off')
    # plt.show()

    outline_mask = np.zeros_like(closed_output, dtype=np.uint8)
    cv2.drawContours(outline_mask, [closed_contour], -1, (1))
    # plt.figure(figsize=(6, 6))
    # plt.imshow(outline_mask, cmap='gray')
    # plt.title("Tissue Outline")
    # plt.axis('off')
    # plt.show()
    # # Optionally, save the result
    # plt.imsave('tissue_outline_mask.png', mask)

else:
    print("No contours found in closed_output")


common_pixels = cv2.bitwise_and(tissue_mask.reshape(tile.ypixels, tile.xpixels).astype(np.uint8), mask.astype(np.uint8))

fig, ax = plt.subplots(1, 4)
# ax[0].imshow(tissue_mask.reshape(14816, 14749), cmap='gray')
# ax[0].set_title('Original tissue mask')
# ax[1].imshow(mask, cmap='gray')
# ax[1].set_title('Tissue segment mask')
ax[2].imshow(outline_mask, cmap='gray')
ax[2].set_title('Tissue outline mask')
ax[3].imshow(common_pixels, cmap='gray')
ax[3].set_title('Cleaned tissue core')
plt.show()

import pyir_image as pir_im
py_image_tool = pir_im.PyIR_Image()
mask_test = py_image_tool.core_mask_cleaner(tissue_mask.reshape(tile.ypixels, tile.xpixels))

si = smartimage.MaskEditor(mask, tile.ypixels, tile.xpixels)
si.start_editing()
# ________________________________________________________________________________________________

