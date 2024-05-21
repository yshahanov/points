import os
import numpy as np
import struct

import random

numpoints = 20000 # [number of points]
max_dist = 15     # [meters]
min_dist = 4      # [meters]

# transform distances to squares (code optimization)
max_dist *= max_dist
min_dist *= min_dist

size_float = 4
size_small_int = 2

# remapping of labels
label_remap = {
0 :  0, # "unlabeled"
1 :  0, # "outlier"
10:  2, # "car"
11:  2, # "bicycle"
13:  2, # "bus"
15:  2, # "motorcycle"
16:  2, # "on-rails"
18:  2, # "truck"
20:  2, # "other-vehicle"
30:  2, # "person"
31:  2, # "bicyclist"
32:  2, # "motorcyclist"
40:  1, # "road"
44:  1, # "parking"
48:  1, # "sidewalk"
49:  1, # "other-ground"
50:  2, # "building"
51:  2, # "fence"
52:  2, # "other-structure"
60:  1, # "lane-marking"
70:  2, # "vegetation"
71:  2, # "trunk"
72:  2, # "terrain"
80:  2, # "pole"
81:  2, # "traffic-sign"
99:  2, # "other-object"
252: 2, # "moving-car"
253: 2, # "moving-bicyclist"
254: 2, # "moving-person"
255: 2, # "moving-motorcyclist"
256: 2, # "moving-on-rails"
257: 2, # "moving-bus"
258: 2, # "moving-truck"
259: 2, # "moving-other-vehicle"
    }

#modify code when at work
dataset_path =  "C:\\Users\\Yaroslav\\Downloads\\semKittiFinal"

def get_filenames_without_extension(directory):
    """Get a set of filenames without their extensions from the given directory."""
    return set(os.path.splitext(filename)[0] for filename in os.listdir(directory))

def delete_unmatched_files(dir1, dir2, ext1, ext2):
    """
    Delete files in both directories that do not have a matching file in the other directory.
    :param dir1: Path to the first directory (e.g., 'path/to/labels')
    :param dir2: Path to the second directory (e.g., 'path/to/point_clouds')
    :param ext1: Extension of the files in the first directory (e.g., '.label')
    :param ext2: Extension of the files in the second directory (e.g., '.bin')
    """
    # Helper function to get filenames without their extensions
    def get_filenames_without_extension(directory, extension):
        return set(os.path.splitext(filename)[0] for filename in os.listdir(directory) if filename.endswith(extension))

    dir1_files = get_filenames_without_extension(dir1, ext1)
    dir2_files = get_filenames_without_extension(dir2, ext2)

    unmatched_in_dir1 = dir1_files - dir2_files
    unmatched_in_dir2 = dir2_files - dir1_files

    for filename in unmatched_in_dir1:
        file_path = os.path.join(dir1, filename + ext1)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    for filename in unmatched_in_dir2:
        file_path = os.path.join(dir2, filename + ext2)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")



dir1 = "C:\\Users\\Yaroslav\\Downloads\\semKittiFinal\\sequences\\00\\labels"  # replace with your actual directory path
dir2 = "C:\\Users\\Yaroslav\\Downloads\\semKittiFinal\\sequences\\00\\velodyne"  # replace with your actual directory path
ext1 = '.label'  # extension of the files in the first directory
ext2 = '.bin'    # extension of the files in the second directory

#print(delete_unmatched_files(dir1, dir2, ext1, ext2))


# semantic kitti data


import struct
import numpy as np


def read_point_cloud(pcpath):
    """
    Reads point cloud data from a ".bin" file.

    INPUT:
        pcpath : path to the point cloud ".bin" file

    RETURNS:
        pointcloud : numpy array of point cloud data
    """
    pointcloud = []

    size_float = 4

    with open(pcpath, "rb") as pc_file:
        byte = pc_file.read(size_float * 4)

        while byte:
            x, y, z, _ = struct.unpack("ffff", byte)  # unpack 4 float values
            pointcloud.append([x, y, z])
            byte = pc_file.read(size_float * 4)

    # Convert list of points to numpy array
    pointcloud = np.array(pointcloud)

    return pointcloud


# Usage example
# pcpath = "C:\\Users\\Yaroslav\\Downloads\\semKittiFinal\\sequences\\00\\velodyne\\002606.bin"
# pointcloud = read_point_cloud(pcpath)
# print("Point Cloud Data:")
# print(len(pointcloud))

def read_labels(labelpath):
    """
    Reads labels from a ".label" file.
    INPUT:
        labelpath : path to the labels ".label" file
    RETURNS:
        labels : numpy array of labels
    """
    labels = []

    size_uint32 = 4

    with open(labelpath, "rb") as label_file:
        label_byte = label_file.read(size_uint32)

        while label_byte:
            label_data = struct.unpack("I", label_byte)[0]  # unpack 1 uint32 value
            label = label_data & 0xFFFF  # Extract lower 16 bits (semantic label)
            labels.append(label)
            label_byte = label_file.read(size_uint32)

    # Convert list of labels to numpy array
    labels = np.array(labels)

    return labels


# Usage example
# labelpath = "C:\\Users\\Yaroslav\\Downloads\\semKittiFinal\\sequences\\00\\labels\\002606.label"
# labels = read_labels(labelpath)
# print("Labels Data:")
# print(len(labels))


def sample(pointcloud, labels, numpoints_to_sample):
    """
        INPUT
            pointcloud          : list of 3D points
            labels              : list of integer labels
            numpoints_to_sample : number of points to sample
    """
    tensor = np.concatenate((pointcloud, np.reshape(labels, (labels.shape[0], 1))), axis= 1)
    tensor = np.asarray(random.choices(tensor, weights=None, cum_weights=None, k=numpoints_to_sample))
    pointcloud_ = tensor[:, 0:3]
    labels_ = tensor[:, 3]
    labels_ = np.array(labels_, dtype=np.int_)
    return pointcloud_, labels_

def readpc(pcpath, labelpath, reduced_labels=True):
    """
    INPUT
        pcpath         : path to the point cloud ".bin" file
        labelpath      : path to the labels ".label" file
        reduced_labels : flag to select which label encoding to return
                        [True]  -> values in range [0, 1, 2]   -- default
                        [False] -> all Semantic-Kitti dataset original labels
    """

    pointcloud, labels = [], []

    with open(pcpath, "rb") as pc_file, open(labelpath, "rb") as label_file:
        byte = pc_file.read(size_float * 4)
        label_byte = label_file.read(size_small_int)
        _ = label_file.read(size_small_int)

        while byte:
            x, y, z, _ = struct.unpack("ffff", byte)  # unpack 4 float values
            label = struct.unpack("H", label_byte)[0]  # unpach 1 Unsigned Short value

            d = x * x + y * y + z * z  # Euclidean norm

            if min_dist < d < max_dist:
                pointcloud.append([x, y, z])
                if reduced_labels:  # for reduced labels range
                    labels.append(label_remap[label])
                else:  # for full labels range
                    labels.append(label)

            byte = pc_file.read(size_float * 4)
            label_byte = label_file.read(size_small_int)
            _ = label_file.read(size_small_int)

    pointcloud = np.array(pointcloud)
    labels = np.array(labels)

    # return fixed_sized lists of points/labels (fixed size: numpoints)
    return sample(pointcloud, labels, numpoints)







