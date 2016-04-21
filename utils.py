"""Provides utility functions for rotation and transformation matrices."""

import numpy as np

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"


def rotx(t):
    """Rotation about the x-axis."""
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])


def roty(t):
    """Rotation about the y-axis."""
    return np.array([[np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rotz(t):
    """Rotation about the z-axis."""
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    t = np.reshape(t, [3, 1])
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
