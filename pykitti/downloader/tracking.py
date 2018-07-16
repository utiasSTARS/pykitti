#!/usr/bin/env python
""""Downloads and unzips the KITTI tracking data.
Warning: This can take a while, and use up >100Gb of disk space."""

from __future__ import print_function

import argparse
import os
import sys

from subprocess import call
import glob

URL_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti/"
tracking_dir_names = ['image_02', 'image_03', 'velodyne', 'calib', 'oxts', 'label_02', 'det_02']
tracking_dir_zip_tags = ['image_2', 'image_3', 'velodyne', 'calib', 'oxts', 'label_2', 'det_2_lsvm']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_root', type=str, default=os.path.join('data', 'kitti'))
    # parser.add_argument('--root', type=str, default=None, help='data folder')

    return parser.parse_args(sys.argv[1:])

## Need to clean up lsvm as their files have trailing whitespaces
def clean_file(filename):
    f = open(filename, 'r')
    new_lines = []
    for line in f.readlines():
        new_lines.append(line.rstrip())
    f.close()

    f = open(filename, 'w')
    for line in new_lines:
        f.write(line + '\n')


    f.close()
def clean_lsvm(lsvm_dir):
    for filename in glob.glob(lsvm_dir + '/*.txt'):
        print('Cleaning ', filename)
        clean_file(filename)


def main():
    args = parse_args()
    kitti_dir = args.kitti_root

    # Perform a few sanity checks to make sure we're operating in the right dir
    # when left with the default args.
    if not os.path.isabs(kitti_dir):
        if not os.path.isdir('src'):
            os.chdir('..')

            if not os.path.isdir('src'):
                print("Please make sure to run this tool from the DynSLAM "
                      "project root when using relative paths.")
                return 1

    tracking_dir = os.path.join(kitti_dir, 'tracking')
    os.makedirs(tracking_dir, exist_ok=True)
    os.chdir(tracking_dir)

    tracking_zip_names = ["data_tracking_" + name + ".zip" for name in tracking_dir_zip_tags]

    for dir_name, zip_name in zip(tracking_dir_names, tracking_zip_names):
        canary_dir = os.path.join('training', dir_name)
        if os.path.isdir(canary_dir):
            print("Directory {} canary dir seems to exist, so I will assume the data is there.".format(canary_dir))
        else:
            if os.path.exists(zip_name):
                print("File {} exists. Not re-downloading.".format(zip_name))
            else:
                url = URL_BASE + zip_name
                print("Downloading file {} to folder {}.".format(zip_name, kitti_dir))
                call(['wget', url])

            call(['unzip', '-o', zip_name])

        if str(canary_dir) == 'training/det_02':
            print("Need to trim whitespaces for lsvm label files")
            clean_lsvm('training/det_02')

    return 0


if __name__ == '__main__':
    sys.exit(main())
