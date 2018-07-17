"""Provides 'tracking', which loads and parses tracking benchmark data."""

import datetime as dt
import glob
import os
from collections import namedtuple
import pandas as pd

import numpy as np

import pykitti.utils as utils
import cv2

try:
    xrange
except NameError:
    xrange = range

__author__ = "Sidney zhang"
__email__ = "sidney@sidazhang.com"


class tracking:
    """Load and parse tracking benchmark data into a usable format."""

    def __init__(self, base_path, sequence, **kwargs):
        """Set the path."""
        self.base_path = base_path
        self.sequence = sequence
        self.frames = kwargs.get('frames', None)

        # Default image file extension is 'png'
        self.imtype = kwargs.get('imtype', 'png')

        # Find all the data files
        self._get_file_lists()
        print('files', len(self.cam2_files))
        # Pre-load data that isn't returned as a generator
        # self._load_calib()

    def __len__(self):
        """Return the number of frames loaded."""
        return len(self.timestamps)

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
        calib_filepath = os.path.join(self.sequence_path + '.txt', 'calib.txt')
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

def to_array_list(df, length=None, by_id=True):
    """Converts a dataframe to a list of arrays, with one array for every unique index entry.
    Index is assumed to be 0-based contiguous. If there is a missing index entry, an empty
    numpy array is returned for it.
    Elements in the arrays are sorted by their id.
    :param df:
    :param length:
    :return:
    """

    if by_id:
        assert 'id' in df.columns

        # if `id` is the only column, don't sort it (and don't remove it)
        if len(df.columns) == 1:
            by_id = False

    idx = df.index.unique()
    if length is None:
        length = max(idx) + 1

    l = [np.empty(0) for _ in xrange(length)]
    for i in idx:
        a = df.loc[i]
        if by_id:
            if isinstance(a, pd.Series):
                a = a[1:]
            else:
                a = a.copy().set_index('id').sort_index()

        l[i] = a.values.reshape((-1, a.shape[-1]))
    return np.asarray(l)


# TODO: Acknowledge this is from HART
class KittiTrackingLabels(object):
    """Kitt Tracking Label parser. It can limit the maximum number of objects per track,
    filter out objects with class "DontCare", or retain only those objects present
    in a given frame.
    """

    columns = 'id class truncated occluded alpha x1 y1 x2 y2 xd yd zd x y z roty score'.split()
    classes = 'Car Van Truck Pedestrian Person_sitting Cyclist Tram Misc DontCare'.split()


    def __init__(self, path_or_df, bbox_with_size=True, remove_dontcare=True, split_on_reappear=True):

        if isinstance(path_or_df, pd.DataFrame):
            self._df = path_or_df
        else:
            if not os.path.exists(path_or_df):
                raise ValueError('File {} doesn\'t exist'.format(path_or_df))

            self._df = pd.read_csv(path_or_df, sep=' ', header=None,
                                   index_col=0, skip_blank_lines=True)

            # Detection files have 1 more column than label files
            # label file has 16 columns
            # detection file has 17 columns (the last column is score)
            # Here it checks the number of columns the df has and sets the
            # column names on the df accordingly
            self._df.columns = self.columns[:len(self._df.columns)]

        self.bbox_with_size = bbox_with_size

        if remove_dontcare:
            self._df = self._df[self._df['class'] != 'DontCare']

        for c in self._df.columns:
            self._convert_type(c, np.float32, np.float64)
            self._convert_type(c, np.int32, np.int64)


        # TODO: Add occlusion filtering back in
        truncated_threshold=(0, 2.)
        occluded_threshold=(0, 3.)
        # if not nest.is_sequence(occluded_threshold):
        #     occluded_threshold = (0, occluded_threshold)
        #
        # if not nest.is_sequence(truncated_threshold):
        #     truncated_threshold = (0, truncated_threshold)
        # TODO: Add occlusion filteringinback in
        # self._df = self._df[self._df['occluded'] >= occluded_threshold[0]]
        # self._df = self._df[self._df['occluded'] <= occluded_threshold[1]]
        #
        # self._df = self._df[self._df['truncated'] >= truncated_threshold[0]]
        # self._df = self._df[self._df['truncated'] <= truncated_threshold[1]]

        # make 0-based contiguous ids
        ids = self._df.id.unique()
        offset = max(ids) + 1
        id_map = {id: new_id for id, new_id in zip(ids, np.arange(offset, len(ids) + offset))}
        self._df.replace({'id': id_map}, inplace=True)
        self._df.id -= offset

        self.ids = list(self._df.id.unique())
        self.max_objects = len(self.ids)
        self.index = self._df.index.unique()

        if split_on_reappear:
            added_ids = self._split_on_reappear(self._df, self.presence, self.ids[-1])
            self.ids.extend(added_ids)
            self.max_objects += len(added_ids)

    def _convert_type(self, column, dest_type, only_from_type=None):
        cond = only_from_type is None or self._df[column].dtype == only_from_type
        if cond:
            self._df[column] = self._df[column].astype(dest_type)

    @property
    def bbox(self):
        bbox = self._df[['id', 'x1', 'y1', 'x2', 'y2']].copy()
        # TODO: Fix this to become x, y, w, h
        if self.bbox_with_size:
            bbox['y2'] -= bbox['y1']
            bbox['x2'] -= bbox['x1']

        """Converts a dataframe to a list of arrays
        :param df:
        :param length:
        :return:
        """

        return to_array_list(bbox)

    @property
    def presence(self):
        return self._presence(self._df, self.index, self.max_objects)

    @property
    def num_objects(self):
        ns = self._df.id.groupby(self._df.index).count()
        absent = list(set(range(len(self))) - set(self.index))
        other = pd.DataFrame([0] * len(absent), absent)
        ns = ns.append(other)
        ns.sort_index(inplace=True)
        return ns.as_matrix().squeeze()

    @property
    def cls(self):
        return to_array_list(self._df[['id', 'class']])

    @property
    def occlusion(self):
        return to_array_list(self._df[['id', 'occluded']])

    @property
    def id(self):
        return to_array_list(self._df['id'])

    def __len__(self):
        return self.index[-1] - self.index[0] + 1

    @classmethod
    def _presence(cls, df, index, n_objects):
        p = np.zeros((index[-1] + 1, n_objects), dtype=bool)
        for i, row in df.iterrows():
            p[i, row.id] = True
        return p

    @classmethod
    def _split_on_reappear(cls, df, p, id_offset):
        """Assign a new identity to an objects that appears after disappearing previously.
        Works on `df` in-place.
        :param df: data frame
        :param p: presence
        :param id_offset: offset added to new ids
        :return:
        """

        next_id = id_offset + 1
        added_ids = []
        nt = p.sum(0)
        start = np.argmax(p, 0)
        end = np.argmax(np.cumsum(p, 0), 0)
        diff = end - start + 1
        is_contiguous = np.equal(nt, diff)
        for id, contiguous in enumerate(is_contiguous):
            if not contiguous:

                to_change = df[df.id == id]
                index = to_change.index
                diff = index[1:] - index[:-1]
                where = np.where(np.greater(diff, 1))[0]
                for w in where:
                    to_change.loc[w + 1:, 'id'] = next_id
                    added_ids.append(next_id)
                    next_id += 1

                df[df.id == id] = to_change

        return added_ids
