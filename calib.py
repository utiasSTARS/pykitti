"""Provides tools to read and processes calibration parameters."""
import os

import numpy as np

import kittitools.utils as utils

author = "Lee Clement"
email = "lee.clement@robotics.utias.utoronto.ca"


def read_calib_file(calibdir, filename):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    filepath = os.path.join(calibdir, filename)
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


def load_calib_rigid(calibdir, filename):
    """Read a rigid transform calibration file as a numpy.matrix."""
    data = read_calib_file(calibdir, filename)
    return utils.transform_from_rot_trans(np.reshape(data['R'], (3, 3)),
                                          np.reshape(data['T'], (3, 1)))


def load_calib_cam_to_cam(calibdir, velo_to_cam_file, cam_to_cam_file):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates
    T_cam0unrect_velo = load_calib_rigid(calibdir, velo_to_cam_file)

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(calibdir, cam_to_cam_file)

    # Create 3x4 projection matrices
    P_rect_00 = np.asmatrix(np.reshape(filedata['P_rect_00'], (3, 4)))
    P_rect_10 = np.asmatrix(np.reshape(filedata['P_rect_01'], (3, 4)))
    P_rect_20 = np.asmatrix(np.reshape(filedata['P_rect_02'], (3, 4)))
    P_rect_30 = np.asmatrix(np.reshape(filedata['P_rect_03'], (3, 4)))

    # Create 4x4 matrix from the rectifying rotation matrix
    R_rect_00 = np.asmatrix(np.eye(4))
    R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))

    # Compute the rectified extrinsics from cam0 to camN
    T0 = np.asmatrix(np.eye(4))
    T0[1, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
    T1 = np.asmatrix(np.eye(4))
    T1[1, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.asmatrix(np.eye(4))
    T2[1, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.asmatrix(np.eye(4))
    T3[1, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = T0 * R_rect_00 * T_cam0unrect_velo
    data['T_cam1_velo'] = T1 * R_rect_00 * T_cam0unrect_velo
    data['T_cam2_velo'] = T2 * R_rect_00 * T_cam0unrect_velo
    data['T_cam3_velo'] = T3 * R_rect_00 * T_cam0unrect_velo

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters
    p_cam = np.matrix([0, 0, 0, 1]).T
    p_velo0 = np.linalg.inv(data['T_cam0_velo']) * p_cam
    p_velo1 = np.linalg.inv(data['T_cam1_velo']) * p_cam
    p_velo2 = np.linalg.inv(data['T_cam2_velo']) * p_cam
    p_velo3 = np.linalg.inv(data['T_cam3_velo']) * p_cam

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data


def load_calib(base_path, date):
    """Load and compute intrinsic and extrinsic calibration parameters."""
    # We'll return the calibration parameters as a dictionary
    data = {}

    # Set the calibration file directory
    calibdir = os.path.join(base_path, date)

    # Load the rigid transformation from velodyne to IMU
    data['T_velo_imu'] = load_calib_rigid(calibdir, 'calib_imu_to_velo.txt')

    # Load the camera intrinsics and extrinsics
    data.update(load_calib_cam_to_cam(
        calibdir, 'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

    # Pre-compute the IMU to rectified camera coordinate transforms
    data['T_cam0_imu'] = data['T_cam0_velo'] * data['T_velo_imu']
    data['T_cam1_imu'] = data['T_cam1_velo'] * data['T_velo_imu']
    data['T_cam2_imu'] = data['T_cam2_velo'] * data['T_velo_imu']
    data['T_cam3_imu'] = data['T_cam3_velo'] * data['T_velo_imu']

    return data
