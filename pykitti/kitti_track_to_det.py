import argparse
import os
import natsort
import numpy as np
import glob
import shutil
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', default='/mnt/sdb1/datasets/kitti', type=str,
                    help='your data folder location')
parser.add_argument('--dest_loc', default='/mnt/sdb1/datasets/kitti_tracking', type=str,
                    help='your data folder location')
args = parser.parse_args()

for folder_type in ['training', 'testing']:
    image2_folders_loc = os.path.join(args.data_loc, folder_type, 'image_02')
    image3_folders_loc = os.path.join(args.data_loc, folder_type, 'image_03')
    velo_folders_loc = os.path.join(args.data_loc, folder_type, 'velodyne')
    image_folders = natsort.natsorted(os.listdir(image2_folders_loc))
    calib_files_folder = os.path.join(args.data_loc, folder_type, 'calib')
    calib_files = natsort.natsorted(os.listdir(calib_files_folder))
    label_files_folder = os.path.join(args.data_loc, 'training', 'label_02')
    label_files = natsort.natsorted(os.listdir(label_files_folder))
    for image_folder in image_folders:
        image2_folder = os.path.join(args.dest_loc, folder_type, image_folder, 'image_2')
        image3_folder = os.path.join(args.dest_loc, folder_type, image_folder, 'image_3')
        velo_folder = os.path.join(args.dest_loc, folder_type, image_folder, 'velodyne')
        Path(image2_folder).mkdir(parents=True, exist_ok=True)
        Path(image3_folder).mkdir(parents=True, exist_ok=True)
        Path(velo_folder).mkdir(parents=True, exist_ok=True)
        imgs = natsort.natsorted(os.listdir(os.path.join(image2_folders_loc, image_folder)))

        print('NOW COPYING ' + folder_type + ' IMAGE_2 FILES FROM ' + image_folder)
        for img_file in glob.iglob(os.path.join(image2_folders_loc, image_folder, '*.png')):
            shutil.copy(img_file, image2_folder)
        print('NOW COPYING ' + folder_type + ' IMAGE_3 FILES FROM ' + image_folder)
        for img_file in glob.iglob(os.path.join(image3_folders_loc, image_folder, '*.png')):
            shutil.copy(img_file, image3_folder)
        print('NOW COPYING ' + folder_type + ' VELODYNE FILES FROM ' + image_folder)
        for velo_file in glob.iglob(os.path.join(velo_folders_loc, image_folder, '*.bin')):
            shutil.copy(velo_file, velo_folder)

        calib_folder = os.path.join(args.dest_loc, folder_type, image_folder, 'calib')
        Path(calib_folder).mkdir(parents=True, exist_ok=True)
        print('NOW COPYING ' + folder_type + ' CALIB FILES FROM ' + image_folder)
        calib_file_read = pd.read_csv(os.path.join(calib_files_folder, image_folder.split('.')[0]+'.txt'),
                                      sep=" ", header=None)
        _calib_file_read = list(calib_file_read.to_numpy())
        calib_file_read = []
        for line in _calib_file_read:
            if line[0] == 'R_rect': line[0] = 'R0_rect:'
            if line[0] == 'Tr_velo_cam': line[0] = 'Tr_velo_to_cam:'
            if line[0] == 'Tr_imu_velo': line[0] = 'Tr_imu_to_velo:'
            if line[0] == 'R0_rect:': calib_file_read.append(np.delete(line, [-1, -2, -3, -4, -5]))
            else: calib_file_read.append(np.delete(line, [-1, -2]))
        file_names = natsort.natsorted(os.listdir(os.path.join(image2_folders_loc, image_folder)))

        for file_name in file_names:
            calib_file_write = open(os.path.join(calib_folder, file_name.split('.')[0] + '.txt'), 'w+')
            for line in calib_file_read:
                for val in line[:-1]:
                    calib_file_write.write(str(val) + ' ')
                calib_file_write.write(str(line[-1]) + '\n')
            calib_file_write.close()

        if folder_type == 'testing': continue
        label_folder = os.path.join(args.dest_loc, folder_type, image_folder, 'label_2')
        Path(label_folder).mkdir(parents=True, exist_ok=True)
        print('NOW COPYING ' + folder_type + ' LABEL FILES FROM ' + image_folder)
        label_file_read = pd.read_csv(os.path.join(label_files_folder, image_folder.split('.')[0] + '.txt'),
                                      sep=" ", header=None)
        label_file_read = list(label_file_read.to_numpy())
        file_names = natsort.natsorted(os.listdir(os.path.join(image2_folders_loc, image_folder)))
        for file_name in file_names:
            frame_num = file_name.split('.')[0]
            label_file_write = open(os.path.join(label_folder, frame_num + '.txt'), 'w+')
            got_frame = False
            for i in range(len(label_file_read)):
                if label_file_read[i][0] == int(frame_num):
                    got_frame = True
                    for val in label_file_read[i][2:-1]:
                        label_file_write.write(str(val) + ' ')
                    label_file_write.write(str(label_file_read[i][-1]) + '\n')
                if got_frame and label_file_read[i][0] != int(frame_num): break
            label_file_write.close()




