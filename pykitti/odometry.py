"""Provides 'odometry', which loads and parses odometry benchmark data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

import pykitti.utils as utils

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


class odometry:
    """Load and parse odometry benchmark data into a usable format."""

    def __init__(self, base_path, sequence, frame_range=None):
        """Set the path."""
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', sequence)
        self.pose_path = os.path.join(base_path, 'poses')
        self.frame_range = frame_range

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join(self.sequence_path, 'calib.txt')
        filedata = utils.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def load_timestamps(self):
        """Load timestamps from file."""
        print('Loading timestamps for sequence ' + self.sequence + '...')

        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            self.timestamps = [self.timestamps[i] for i in self.frame_range]

        print('Found ' + str(len(self.timestamps)) + ' timestamps...')

        print('done.')

    def load_poses(self):
        """Load ground truth poses from file."""
        print('Loading poses for sequence ' + self.sequence + '...')

        pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

        # Read and parse the poses
        try:
            self.T_w_cam0 = []
            with open(pose_file, 'r') as f:
                for line in f.readlines():
                    T = np.fromstring(line, dtype=float, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    self.T_w_cam0.append(T)
            print('done.')

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence ' +
                  self.sequence + '.')

    def load_gray(self, **kwargs):
        """Load monochrome stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 for
        easy use with OpenCV.
        """
        print('Loading monochrome images from sequence ' +
              self.sequence + '...')

        imL_path = os.path.join(self.sequence_path, 'image_0', '*.png')
        imR_path = os.path.join(self.sequence_path, 'image_1', '*.png')

        imL_files = sorted(glob.glob(imL_path))
        imR_files = sorted(glob.glob(imR_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            imL_files = [imL_files[i] for i in self.frame_range]
            imR_files = [imR_files[i] for i in self.frame_range]

        print('Found ' + str(len(imL_files)) + ' image pairs...')

        self.gray = utils.load_stereo_pairs(imL_files, imR_files, **kwargs)

        print('done.')

    def load_rgb(self, **kwargs):
        """Load RGB stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 and BGR for
        easy use with OpenCV.
        """
        print('Loading color images from sequence ' +
              self.sequence + '...')

        imL_path = os.path.join(self.sequence_path, 'image_2', '*.png')
        imR_path = os.path.join(self.sequence_path, 'image_3', '*.png')

        imL_files = sorted(glob.glob(imL_path))
        imR_files = sorted(glob.glob(imR_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            imL_files = [imL_files[i] for i in self.frame_range]
            imR_files = [imR_files[i] for i in self.frame_range]

        print('Found ' + str(len(imL_files)) + ' image pairs...')

        self.rgb = utils.load_stereo_pairs(imL_files, imR_files, **kwargs)

        print('done.')

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.sequence_path, 'velodyne', '*.bin')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            velo_files = [velo_files[i] for i in self.frame_range]

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')

        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = utils.load_velo_scans(velo_files)

        print('done.')
