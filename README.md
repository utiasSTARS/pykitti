# pykitti
This package provides a minimal set of tools for working with the KITTI dataset [[1]](#references) in Python. So far only the raw datasets and associated calibration data are supported.

## Setup
To get started, simply clone the repository to your machine
```
git clone https://github.com/utiasSTARS/pykitti.git
```
and run the provided setup tool
```
cd pykitti
python setup.py install
```

## Notation
Homogeneous coordinate transformations are provided as 4x4 `numpy.array` objects and are denoted as `T_destinationFrame_originFrame`.

Pinhole camera intrinsics for camera `N` are provided as 3x3 `numpy.array` objects and are denoted as `K_camN`. Stereo pair baselines are given in meters as `b_gray` for the monochrome stereo pair (`cam0` and `cam1`), and `b_rgb` for the color stereo pair (`cam2` and `cam3`).

## Example
More detailed examples can be found in the `tests` directory, but the general idea is to specify what dataset you want to load, then load the parts you need and do something with them:

```python
import pykitti

basedir = '/your/dataset/dir'
date = '2011_09_26'
drive = '0019'

# The range argument is optional - default is None, which loads the whole dataset
data = pykitti.raw(basedir, date, drive, range(0, 50, 5))

# Sensor calibration data are loaded automatically
point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

# Other data are loaded only if requested
data.load_oxts()
point_w = data.oxts[0].T_w_imu.dot(point_imu)

data.load_rgb()
cam2_image = data.rgb[0].left
```

### OpenCV
Image data can be automatically converted to an OpenCV-friendly format (i.e., `uint8` with `BGR` color channel ordering) simply by specifying an additional parameter in the image loader function:

```python
data.load_gray(format='cv2')  # Loads images as uint8 grayscale
data.load_rgb(format='cv2')   # Loads images as uint8 with BGR ordering
```

Note: This package does not actually require that OpenCV be installed on your system, except to run `test_raw_cv2.py`.

## References
[1] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision meets robotics: The KITTI dataset," Int. J. Robot. Research (IJRR), vol. 32, no. 11, pp. 1231–1237, Sep. 2013. [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
