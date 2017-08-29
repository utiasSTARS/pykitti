"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
# dataset = pykitti.raw(basedir, date, drive)
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 20, 5))

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of datetime objects
# dataset.oxts:       Generator to load OXTS packets as named tuples
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
second_pose = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu
first_gray = next(iter(dataset.gray))
first_cam1 = next(iter(dataset.cam1))
first_rgb = next(iter(dataset.rgb))
first_cam2 = next(iter(dataset.cam2))
third_velo = next(iter(itertools.islice(dataset.velo, 2, None)))

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nDrive: ' + str(dataset.drive))
print('\nFrame range: ' + str(dataset.frames))

print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond IMU pose:\n' + str(second_pose))

f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(first_gray[0], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(first_cam1, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(first_cam2)
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(first_rgb[1])
ax[1, 1].set_title('Right RGB Image (cam3)')


f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range = range(0, third_velo.shape[0], 100)
ax2.scatter(third_velo[velo_range, 0],
            third_velo[velo_range, 1],
            third_velo[velo_range, 2],
            c=third_velo[velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

plt.show()
