"""Example of kitti.raw usage with OpenCV."""
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import kittitools as kitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/Users/leeclement/Desktop/datasets'

# Specify the dataset to load
date = '2011_09_26'
drive = '0019'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
dataset = kitti.raw(basedir, date, drive, frame_range)

# Load image data
dataset.load_gray(opencv=True)  # Loads as uint8
dataset.load_rgb(opencv=True)   # Loads as uint8 with BGR ordering

# Do some stereo processing
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disp_gray = stereo.compute(dataset.gray[0]['left'], dataset.gray[0]['right'])
disp_rgb = stereo.compute(
    cv2.cvtColor(dataset.rgb[0]['left'], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(dataset.rgb[0]['right'], cv2.COLOR_BGR2GRAY))

# Display some data
f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(dataset.gray[0]['left'], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(disp_gray, cmap='viridis')
ax[0, 1].set_title('Gray Stereo Disparity')

ax[1, 0].imshow(cv2.cvtColor(dataset.rgb[0]['left'], cv2.COLOR_BGR2RGB))
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(disp_rgb, cmap='viridis')
ax[1, 1].set_title('RGB Stereo Disparity')

plt.show()
