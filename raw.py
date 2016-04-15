"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os

import matplotlib.image as mpimg
import numpy as np

import kittitools.calib as calib
import kittitools.utils as utils

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

    def __get_poses_from_oxts(self, oxts):
        er = 6378137  # earth radius (approx.) in meters

        t_0 = []  # initial position
        for o in oxts:
            # Use a Mercator projection to get the translation vector
            scale = np.cos(o['lat'] * np.pi / 180.0)
            tx = scale * o['lon'] * np.pi * er / 180.0
            ty = scale * np.log(np.tan((90.0 + o['lat']) * np.pi / 360.0))
            tz = o['alt']
            t = np.matrix([tx, ty, tz]).T

            # We want the initial position to be the origin, but keep the ENU
            # coordinate system
            if len(t_0) == 0:
                t_0 = t

            # Use the Euler angles to get the rotation matrix
            rx = o['roll']
            ry = o['pitch']
            rz = o['yaw']
            R = utils.rotz(rz) * utils.roty(ry) * utils.rotx(rx)

            # Combine the translation and rotation into a homogeneous transform
            o['T_imu_w'] = utils.transform_from_rot_trans(R, t - t_0)

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
        self.oxts = []
        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(x) for x in line[-5:]]

                    # Per dataformat.txt
                    data = {'lat': line[0], 'lon': line[1], 'alt': line[2],
                            'roll': line[3], 'pitch': line[4], 'yaw': line[5],
                            'vn': line[6], 've': line[7],
                            'vf': line[8], 'vl': line[9], 'vu': line[10],
                            'ax': line[11], 'ay': line[12], 'az': line[13],
                            'af': line[14], 'al': line[15], 'au': line[16],
                            'wx': line[17], 'wy': line[18], 'wz': line[19],
                            'wf': line[20], 'wl': line[21], 'wu': line[22],
                            'pos_accuracy': line[23], 'vel_accuracy': line[24],
                            'navstat': line[25], 'numsats': line[26],
                            'posmode': line[27], 'velmode': line[28],
                            'orimode': line[29]}

                    self.oxts.append(data)

        # Precompute the IMU poses in the world frame and add them to the
        # OXTS dictionary
        self.__get_poses_from_oxts(self.oxts)

        print('done.')

    def __load_stereo(self, imL_path, imR_path, opencv):
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
        impairs = []
        for imfiles in zip(imL_files, imR_files):
            # Convert to uint8 for OpenCV if requested
            if opencv:
                impairs.append({
                    'left': np.uint8(mpimg.imread(imfiles[0]) * 255),
                    'right': np.uint8(mpimg.imread(imfiles[1]) * 255)})
            else:
                impairs.append({'left': mpimg.imread(imfiles[0]),
                                'right': mpimg.imread(imfiles[1])})

        return impairs

    def load_gray(self, opencv=False):
        """Load monochrome stereo images from file.

        Setting opencv=True will convert the images to uint8 for
        easy use with OpenCV.
        """
        print('Loading monochrome images from ' + self.drive + '...')

        imL_path = os.path.join(self.path, 'image_00')
        imR_path = os.path.join(self.path, 'image_01')

        self.gray = self.__load_stereo(imL_path, imR_path, opencv)

        print('done.')

    def load_rgb(self, opencv=False):
        """Load RGB stereo images from file.

        Setting opencv=True will convert the images to uint8 and BGR for
        easy use with OpenCV.
        """
        print('Loading color images from ' + self.drive + '...')

        imL_path = os.path.join(self.path, 'image_02')
        imR_path = os.path.join(self.path, 'image_03')

        self.rgb = self.__load_stereo(imL_path, imR_path, opencv)

        # Convert from RGB to BGR for OpenCV if requested
        if opencv:
            for pair in self.rgb:
                pair['left'] = pair['left'][:, :, ::-1]
                pair['right'] = pair['right'][:, :, ::-1]

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
