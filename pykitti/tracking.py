"""Provides 'tracking', which loads and parses tracking benchmark data."""

import glob
import os
from collections import namedtuple, defaultdict
import pandas as pd
import numpy as np

import pykitti.utils as utils

try:
    xrange
except NameError:
    xrange = range


class tracking:
    """Load and parse tracking benchmark data into a usable format."""

    def __init__(self, base_path, sequence, **kwargs):
        """Set the path."""
        self.base_path = base_path
        self.sequence = sequence
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Exclude DontCare objects from dataset
        self.ignore_dontcare = kwargs.get('ignore_dontcare', False)

        # Find all the data files
        self._get_file_lists()

        # Pre-load data that isn't returned as a generator
        self._load_calib()
        self._load_objects()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.cam2_files)

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
    
    def get_objects(self, idx):
        """Return a list of objects visible at the specified index."""
        return self.objects[idx]
    

    def _get_file_lists(self):
        """Find and list data files for each sensor."""
        self.cam2_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_02',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.cam3_files = sorted(glob.glob(
            os.path.join(self.base_path,
                         'image_03',
                         self.sequence,
                         '*.{}'.format(self.imtype))))
        self.velo_files = sorted(glob.glob(
            os.path.join(self.base_path,
                        'velodyne',
                        self.sequence,
                         '*.bin')))

        # Subselect the chosen range of frames, if any
        if self.frames is not None:
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
        calib_filepath = os.path.join(
            self.base_path, 'calib/{}.txt'.format(self.sequence))
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

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect'], (3, 3))
        data['R_rect_00'] = R_rect_00

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # # Compute the velodyne to rectified camera coordinate transforms
        T_cam_velo = filedata['Tr_velo_cam'].reshape((3, 4))
        T_cam_velo = np.vstack([T_cam_velo, [0, 0, 0, 1]])
        data['T_cam0_velo'] = R_rect_00.dot(T_cam_velo)
        data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_cam_velo))
        data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_cam_velo))
        data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_cam_velo))

        # # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        self.calib = namedtuple('CalibData', data.keys())(*data.values())
    

    def _load_objects(self):
        """Parse object tracklets"""

        # Load tracklet data as a pandas dataframe
        track_filename = os.path.join(
            self.base_path, 'label_02/{}.txt'.format(self.sequence))
        columns = ('fid tid type trunc occ alpha x1 y1 x2 y2 '
                   'h w l x y z ry score').split()
        df = pd.read_csv(track_filename, sep=' ', header=None, names=columns,
                         index_col=None, skip_blank_lines=True)
        
        self.objects = [list() for _ in self.cam2_files]

        # Iterate over the entries in the csv file
        for data in df.itertuples(index=False):

            # Skip DontCare objects if requested
            if self.ignore_dontcare and data.type == 'DontCare':
                continue
            
            # Extract 2D bounding box, dimensions and location from dataframe
            bbox = utils.BoundingBox(data.x1, data.y1, data.x2, data.y2)
            dims = utils.Dimensions(data.l, data.h, data.w)
            loc = utils.Location(data.x, data.y, data.z)

            # Construct object data
            obj_data = utils.ObjectData(
                data.tid, data.type, data.trunc, data.occ, data.alpha, 
                bbox, dims, loc, data.ry, data.score
            )

            # Assign object to frame with index fid
            if self.frames is not None:
                if data.fid in self.frames:
                    index = self.frames.index(data.fid)
                    self.objects[index].append(obj_data)
            else:
                self.objects[data.fid].append(obj_data)
    

    def _load_oxts(self):
        """Load OXTS data from file."""
        oxt_filename = os.path.join(
            self.base_path, 'oxts/{}.txt'.format(self.sequence))
        self.oxts = utils.load_oxts_packets_and_poses([oxt_filename])
