"""Example of pykitti.raw usage with OpenCV."""
import cv2
import matplotlib.pyplot as plt

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/Users/leeclement/Desktop/KITTI/raw'

# Specify the dataset to load
date = '2011_09_30'
drive = '0016'

# Load the data. Optionally, specify the frame range to load.
# Passing imformat='cv2' will convert images to uint8 and BGR for
# easy use with OpenCV.
dataset = pykitti.raw(basedir, date, drive,
                      frames=range(0, 20, 5), imformat='cv2')

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of datetime objects
# dataset.oxts:       Generator to load OXTS packets as named tuples
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
first_gray = next(iter(dataset.gray))
first_rgb = next(iter(dataset.rgb))

# Do some stereo processing
stereo = cv2.StereoBM_create()
disp_gray = stereo.compute(first_gray[0], first_gray[1])
disp_rgb = stereo.compute(
    cv2.cvtColor(first_rgb[0], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(first_rgb[1], cv2.COLOR_BGR2GRAY))

# Display some data
f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(first_gray[0], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(disp_gray, cmap='viridis')
ax[0, 1].set_title('Gray Stereo Disparity')

ax[1, 0].imshow(cv2.cvtColor(first_rgb[0], cv2.COLOR_BGR2RGB))
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(disp_rgb, cmap='viridis')
ax[1, 1].set_title('RGB Stereo Disparity')

plt.show()
