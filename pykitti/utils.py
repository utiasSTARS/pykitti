"""Provides helper methods for loading and parsing KITTI data."""

from collections import namedtuple

import matplotlib.image as mpimg
import numpy as np

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def load_stereo_pairs(imL_files, imR_files, **kwargs):
    """Helper method to read stereo image pairs."""
    StereoPair = namedtuple('StereoPair', 'left, right')

    impairs = []
    for imfiles in zip(imL_files, imR_files):
        # Convert to uint8 and BGR for OpenCV if requested
        imformat = kwargs.get('format', '')
        if imformat is 'cv2':
            imL = np.uint8(mpimg.imread(imfiles[0]) * 255)
            imR = np.uint8(mpimg.imread(imfiles[1]) * 255)

            # Convert RGB to BGR
            if len(imL.shape) > 2:
                imL = imL[:, :, ::-1]
                imR = imR[:, :, ::-1]

        else:
            imL = mpimg.imread(imfiles[0])
            imR = mpimg.imread(imfiles[1])

        impairs.append(StereoPair(imL, imR))

    return impairs


def load_velo_scans(velo_files):
    """Helper method to parse velodyne binary files into a list of scans."""
    scan_list = []
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)
        scan_list.append(scan.reshape((-1, 4)))

    return scan_list
