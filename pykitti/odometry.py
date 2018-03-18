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

    def __init__(self, base_path, sequence, **kwargs):
        """Set the path."""
        self.sequence = sequence
        self.sequence_path = os.path.join(base_path, 'sequences', sequence)
        self.pose_path = os.path.join(base_path, 'poses')
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_timestamps()
        self._load_poses()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

    @property
    def cam0(self):
        """Generator to read image files for cam0 (monochrome left)."""
        return utils.yield_images(self.cam0_files, mode='L')

    def get_cam0(self, idx):
        """Read image file for cam0 (monochrome left) at the specified index."""
        return utils.load_image(self.cam0_files[idx], mode='L')

    @property
    def cam1(self):
        """Generator to read image files for cam1 (monochrome right)."""
        return utils.yield_images(self.cam1_files, mode='L')

    def get_cam1(self, idx):
        """Read image file for cam1 (monochrome right) at the specified index."""
        return utils.load_image(self.cam1_files[idx], mode='L')

    @property
    def cam2(self):
        """Generator to read image files for cam2 (RGB left)."""
        return utils.yield_images(self.cam2_files, mode='RGB')

    def get_cam2(self, idx):
        """Read image file for cam2 (RGB left) at the specified index."""
        return utils.load_image(self.cam2_files[idx], mode='RGB')

    @property
    def cam3(self):
        """Generator to read image files for cam0 (RGB right)."""
        return utils.yield_images(self.cam3_files, mode='RGB')

    def get_cam3(self, idx):
        """Read image file for cam3 (RGB right) at the specified index."""
        return utils.load_image(self.cam3_files[idx], mode='RGB')

    @property
    def gray(self):
        """Generator to read monochrome stereo pairs from file.
        """
        return zip(self.cam0, self.cam1)

    def get_gray(self, idx):
        """Read monochrome stereo pair at the specified index."""
        return (self.get_cam0(idx), self.get_cam1(idx))

    @property
    def rgb(self):
        """Generator to read RGB stereo pairs from file.
        """
        return zip(self.cam2, self.cam3)

    def get_rgb(self, idx):
        """Read RGB stereo pair at the specified index."""
        return (self.get_cam2(idx), self.get_cam3(idx))

    @property
    def velo(self):
        """Generator to read velodyne [x,y,z,reflectance] scan data from binary files."""
        # Return a generator yielding Velodyne scans.
        # Each scan is a Nx4 array of [x,y,z,reflectance]
        return utils.yield_velo_scans(self.velo_files)

    def get_velo(self, idx):
        """Read velodyne [x,y,z,reflectance] scan at the specified index."""
        return utils.load_velo_scan(self.velo_files[idx])

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam0_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_0',
                         '*.{}'.format(self.imtype))))
        self.cam1_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_1',
                         '*.{}'.format(self.imtype))))
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_2',
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'image_3',
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.sequence_path, 'velodyne',
                         '*.bin')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.cam0_files = utils.subselect_files(
                self.cam0_files, self.frames)
            self.cam1_files = utils.subselect_files(
                self.cam1_files, self.frames)
            self.cam2_files = utils.subselect_files(
                self.cam2_files, self.frames)
            self.cam3_files = utils.subselect_files(
                self.cam3_files, self.frames)
            self.velo_files = utils.subselect_files(
                self.velo_files, self.frames)

    def _load_calib(self):
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

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

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

    def _load_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(self.sequence_path, 'times.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                t = dt.timedelta(seconds=float(line))
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
            self.timestamps = [self.timestamps[i] for i in self.frames]

    def _load_poses(self):
        """Load ground truth poses (T_w_cam0) from file."""
        pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                if self.frames is not None:
                    lines = [lines[i] for i in self.frames]

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence ' +
                  self.sequence + '.')

        self.poses = poses
