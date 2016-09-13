"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

import pykitti.utils as utils

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


class raw:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, frame_range=None):
        """Set the path."""
        self.drive = date + '_drive_' + drive + '_sync'
        self.calib_path = os.path.join(base_path, date)
        self.data_path = os.path.join(base_path, date, self.drive)
        self.frame_range = frame_range

    def _load_calib_rigid(self, filename):
        """Read a rigid transform calibration file as a numpy.array."""
        filepath = os.path.join(self.calib_path, filename)
        data = utils.read_calib_file(filepath)
        return utils.transform_from_rot_trans(data['R'], data['T'])

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
        # We'll return the camera calibration as a dictionary
        data = {}

        # Load the rigid transformation from velodyne coordinates
        # to unrectified cam0 coordinates
        T_cam0unrect_velo = self._load_calib_rigid(velo_to_cam_file)

        # Load and parse the cam-to-cam calibration data
        cam_to_cam_filepath = os.path.join(self.calib_path, cam_to_cam_file)
        filedata = utils.read_calib_file(cam_to_cam_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
        P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

        # Create 4x4 matrix from the rectifying rotation matrix
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))

        # Compute the rectified extrinsics from cam0 to camN
        T0 = np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
        data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
        data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
        data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))

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

        return data

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the rigid transformation from velodyne to IMU
        data['T_velo_imu'] = self._load_calib_rigid('calib_imu_to_velo.txt')

        # Load the camera intrinsics and extrinsics
        data.update(self._load_calib_cam_to_cam(
            'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

        # Pre-compute the IMU to rectified camera coordinate transforms
        data['T_cam0_imu'] = data['T_cam0_velo'].dot(data['T_velo_imu'])
        data['T_cam1_imu'] = data['T_cam1_velo'].dot(data['T_velo_imu'])
        data['T_cam2_imu'] = data['T_cam2_velo'].dot(data['T_velo_imu'])
        data['T_cam3_imu'] = data['T_cam3_velo'].dot(data['T_velo_imu'])

        self.calib = namedtuple('CalibData', data.keys())(*data.values())

    def load_timestamps(self):
        """Load timestamps from file."""
        print('Loading OXTS timestamps from ' + self.drive + '...')

        timestamp_file = os.path.join(
            self.data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                self.timestamps.append(t)

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            self.timestamps = [self.timestamps[i] for i in self.frame_range]

        print('Found ' + str(len(self.timestamps)) + ' timestamps...')

        print('done.')

    def _poses_from_oxts(self, oxts_packets):
        """Helper method to compute SE(3) pose matrices from OXTS packets."""
        er = 6378137.  # earth radius (approx.) in meters

        # compute scale from first lat value
        scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

        t_0 = []    # initial position
        poses = []  # list of poses computed from oxts
        for packet in oxts_packets:
            # Use a Mercator projection to get the translation vector
            tx = scale * packet.lon * np.pi * er / 180.
            ty = scale * er * \
                np.log(np.tan((90. + packet.lat) * np.pi / 360.))
            tz = packet.alt
            t = np.array([tx, ty, tz])

            # We want the initial position to be the origin, but keep the ENU
            # coordinate system
            if len(t_0) == 0:
                t_0 = t

            # Use the Euler angles to get the rotation matrix
            Rx = utils.rotx(packet.roll)
            Ry = utils.roty(packet.pitch)
            Rz = utils.rotz(packet.yaw)
            R = Rz.dot(Ry.dot(Rx))

            # Combine the translation and rotation into a homogeneous transform
            poses.append(utils.transform_from_rot_trans(R, t - t_0))

        return poses

    def load_oxts(self):
        """Load OXTS data from file."""
        print('Loading OXTS data from ' + self.drive + '...')

        # Find all the data files
        oxts_path = os.path.join(self.data_path, 'oxts', 'data', '*.txt')
        oxts_files = sorted(glob.glob(oxts_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            oxts_files = [oxts_files[i] for i in self.frame_range]

        print('Found ' + str(len(oxts_files)) + ' OXTS measurements...')

        # Extract the data from each OXTS packet
        # Per dataformat.txt
        OxtsPacket = namedtuple('OxtsPacket',
                                'lat, lon, alt, ' +
                                'roll, pitch, yaw, ' +
                                'vn, ve, vf, vl, vu, ' +
                                'ax, ay, az, af, al, au, ' +
                                'wx, wy, wz, wf, wl, wu, ' +
                                'pos_accuracy, vel_accuracy, ' +
                                'navstat, numsats, ' +
                                'posmode, velmode, orimode')

        oxts_packets = []
        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    data = OxtsPacket(*line)
                    oxts_packets.append(data)

        # Precompute the IMU poses in the world frame
        T_w_imu = self._poses_from_oxts(oxts_packets)

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
        self.oxts = []
        for (p, T) in zip(oxts_packets, T_w_imu):
            self.oxts.append(OxtsData(p, T))

        print('done.')

    def load_gray(self, **kwargs):
        """Load monochrome stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 for
        easy use with OpenCV.
        """
        print('Loading monochrome images from ' + self.drive + '...')

        imL_path = os.path.join(self.data_path, 'image_00', 'data', '*.png')
        imR_path = os.path.join(self.data_path, 'image_01', 'data', '*.png')

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
        print('Loading color images from ' + self.drive + '...')

        imL_path = os.path.join(self.data_path, 'image_02', 'data', '*.png')
        imR_path = os.path.join(self.data_path, 'image_03', 'data', '*.png')

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
        velo_path = os.path.join(
            self.data_path, 'velodyne_points', 'data', '*.bin')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            velo_files = [velo_files[i] for i in self.frame_range]

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')

        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = utils.load_velo_scans(velo_files)

        print('done.')
