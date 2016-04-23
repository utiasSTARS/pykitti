"""Provides 'Raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import matplotlib.image as mpimg
import numpy as np

import pykitti.calib as calib
import pykitti.utils as utils

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


class raw:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, frame_range=None):
        """Set the path."""
        self.drive = date + '_drive_' + drive + '_sync'
        self.path = os.path.join(base_path, date, self.drive)
        self.frame_range = frame_range
        self.calib = calib.load_calib(base_path, date)

    def load_timestamps(self):
        """Load timestamps from file."""
        print('Loading OXTS timestamps from ' + self.drive + '...')

        timestamps_path = os.path.join(
            self.path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        self.timestamps = []
        with open(timestamps_path, 'r') as f:
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

    def __get_poses_from_oxts(self, oxts_packets):
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
        oxts_path = os.path.join(self.path, 'oxts', 'data', '*.txt')
        oxts_files = glob.glob(oxts_path)

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
                    line[-5:] = [int(x) for x in line[-5:]]

                    data = OxtsPacket(*line)
                    oxts_packets.append(data)

        # Precompute the IMU poses in the world frame
        T_w_imu = self.__get_poses_from_oxts(oxts_packets)

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
        self.oxts = []
        for (p, T) in zip(oxts_packets, T_w_imu):
            self.oxts.append(OxtsData(p, T))

        print('done.')

    def __load_stereo(self, imL_path, imR_path, **kwargs):
        # Find all the image files
        imdataL_path = os.path.join(imL_path, 'data', '*.png')
        imdataR_path = os.path.join(imR_path, 'data', '*.png')
        imL_files = glob.glob(imdataL_path)
        imR_files = glob.glob(imdataR_path)

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            imL_files = [imL_files[i] for i in self.frame_range]
            imR_files = [imR_files[i] for i in self.frame_range]

        print('Found ' + str(len(imL_files)) + ' image pairs...')

        # Read all the image files
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

    def load_gray(self, **kwargs):
        """Load monochrome stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 for
        easy use with OpenCV.
        """
        print('Loading monochrome images from ' + self.drive + '...')

        imL_path = os.path.join(self.path, 'image_00')
        imR_path = os.path.join(self.path, 'image_01')

        self.gray = self.__load_stereo(imL_path, imR_path, **kwargs)

        print('done.')

    def load_rgb(self, **kwargs):
        """Load RGB stereo images from file.

        Setting imformat='cv2' will convert the images to uint8 and BGR for
        easy use with OpenCV.
        """
        print('Loading color images from ' + self.drive + '...')

        imL_path = os.path.join(self.path, 'image_02')
        imR_path = os.path.join(self.path, 'image_03')

        self.rgb = self.__load_stereo(imL_path, imR_path, **kwargs)

        print('done.')

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.path, 'velodyne_points', 'data', '*.bin')
        velo_files = glob.glob(velo_path)

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            velo_files = [velo_files[i] for i in self.frame_range]

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')

        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = []
        for filename in velo_files:
            scan = np.fromfile(filename, dtype=np.float32)
            self.velo.append(scan.reshape((-1, 4)))

        print('done.')
