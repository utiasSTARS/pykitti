import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pykitti

basedir = 'raw'
date = '2011_09_26'
drive = '0001'

data = pykitti.raw(basedir, date, drive)

"""
The coordinate systems of pointcloud:

* ImageX:     Each point's corresponding pixel coordinate. eg. z_i = [u,v,1]
* CameraX:    Pointcloud coordinate at four cameras (0,1,2,3). eg. y_i = [x,y,z,1]
* Reference:  Equal to Camera0. eg. y_0 = [x,y,z,1]
* Rectified:  Aligned coordinate for the four cameras. eg. r = [x,y,z,1]
* Velodyne:   Original velodyne coordinate. eg. v = [x,y,z,r]
* IMU/GPS:    Pointcloud coordinate at the IMU device. eg. m = [x,y,z,1]
* World:      Real world coordinate, eg. w = [x,y,z,1]

Pykitti transformation matrices:

data.calib.T_camX_velo:  Velodyne -> CameraX
data.calib.K_camX:       CameraX -> ImageX
data.calib.T_velo_imu:   Velodyne -> IMU
data.oxts[idx].T_w_imu:  IMU -> World
"""

# function to draw pointcloud
def draw_point_cloud(ax, pointcloud, axes=[0,1,2], keep_ratio=0.5, pointsize=0.05, color='k'):
    #axes_limits = [[-20, 80], [-20, 20], [-3, 10]]
    axes_limits = [[-10, 70], [-40, 40], [-2, 15]]

    nPoints = pointcloud.shape[0]
    nSample = int(nPoints * keep_ratio)
    if nPoints == 0:
        return

    indices = np.random.choice(nPoints, nSample)

    if len(axes)==3: 
        ax.axis('off')
    
    if not isinstance(color, str):
        color = color[indices]

    ax.scatter(*np.transpose(pointcloud[indices[:, None], axes]), s=pointsize, c=color, alpha=0.5)
    ax.set_xlabel('{} axis'.format(axes[0]))
    ax.set_ylabel('{} axis'.format(axes[1]))

    if len(axes)==3:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.view_init(50, 135)
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])

# function to project velo to camera2, then get the rgb information of velo.
def get_rgb_velo(data, i):
    img = np.asarray(data.get_cam2(i)) # PIL image -> [H=375, W=1242, 3]
    img = img / 255.0
    velo = data.get_velo(i).T # [4, N]
    # velo -> cam2
    cam2 = data.calib.T_cam2_velo @ velo # [4=<x, y, z, 0>, N] 
    cam2 = cam2[:3] / cam2[2] # [x, y, z, 0] -> [x/z, y/z, 1]
    uv = data.calib.K_cam2 @ cam2 
    uv = uv[:2].astype(np.int32) # [2, N]
    # mask out points outside the image
    mask_x = np.logical_and(uv[0]<img.shape[1], uv[0]>=0)
    mask_y = np.logical_and(uv[1]<img.shape[0], uv[1]>=0)
    mask = np.logical_and(mask_x, mask_y)
    uv = uv[:, mask]
    rgb = img[uv[1], uv[0]].T # [3, N']
    velo = velo[:, mask]

    return velo, rgb

# function to plot one frame's velo in Velo coordinate.
def plot_velo(data, i, axes=[0,1,2], keep_ratio=0.1, pointsize=0.1, use_rgb=False):
    if use_rgb:
        velo, rgb = get_rgb_velo(data, i)
        color = rgb.T
    else:
        velo = data.get_velo(i).T
        color = 'k'

    f = plt.figure()
    if len(axes) == 3:
        ax = f.add_subplot(111, projection='3d')
    else:
        ax = f.add_subplot(111)

    draw_point_cloud(ax, velo.T[:,:3], axes, keep_ratio=keep_ratio, pointsize=pointsize, color=color)
    plt.show()

# function to plot all the velos in World coordinate.
def plot_velo_frames(data, axes=[0,1,2], keep_ratio=0.001, pointsize=0.1, use_rgb=False):
    f = plt.figure()
    if len(axes) == 3:
        ax = f.add_subplot(111, projection='3d')
    else:
        ax = f.add_subplot(111)
    
    for i in range(len(data)):
        if use_rgb:
            velo, rgb = get_rgb_velo(data, i)
            color = rgb.T
        else:
            velo = data.get_velo(i).T # [4, N]
            color = 'k'

        velo = data.calib.T_velo_imu @ velo # Velo -> IMU
        velo = data.oxts[i].T_w_imu @ velo # IMU -> World
        
        draw_point_cloud(ax, velo.T[:,:3], axes, keep_ratio=keep_ratio, pointsize=pointsize, color=color)

    plt.show()


if __name__ == "__main__":
    plot_velo(data, 0, use_rgb=True)
    plot_velo_frames(data, axes=[0,1], keep_ratio=0.01, use_rgb=True)

