## Synopsis
This package provides a minimal set of tools for working with the KITTI dataset [1] in Python. So far only the raw datasets and associated calibration data are supported.

## Notation
Homogeneous coordinate transformations are provided as 4x4 `numpy.matrix` objects and are denoted as `T_destinationFrame_originFrame`.

Pinhole camera intrinsics for camera `N` are provided as 3x3 `numpy.matrix` objects and are denoted as `K_camN`. Stereo pair baselines are given in meters as
`b_gray` for the monochrome stereo pair (`cam0` and `cam1`), and `b_rgb` for the color stereo pair (`cam2` and `cam3`).

## Example
More detailed examples can be found in the `tests` directory, but the general idea is to specify what dataset you want to load, then load the parts you need and do something with them:
```python
import kittitools

basedir = '/your/dataset/dir'
date = '2011_09_26'
drive = '0019'

# The range argument is optional - leaving it empty loads the whole dataset
data = kittitools.raw(basedir, date, drive, range(0, 50, 5))

# Sensor calibration data are loaded automatically
point_cam0 = data.calib['T_cam0_velo'] * point_velo

# Other data are loaded only if requested
data.load_oxts()
point_imu = data.oxts[0]['T_imu_w'] * point_w

data.load_rgb()
cam2_image = data.rgb[0]['left']
```

## References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. http://www.cvlibs.net/datasets/kitti/
