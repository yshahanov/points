import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# do the same with other file format

directory = ""

# all_coordinates: list of lists (coordinates x y z)
# directory: path to csv files

def extract_from_csv(all_coordinates, directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, usecols=[0, 1, 2])
            coordinates = df.to_numpy().tolist()
            all_coordinates.extend(coordinates)
    return all_coordinates # in perspective should return a data
                           # structure that could be used for training a model


# list of lists ... could be modified in the future
def visualize_3d(coordinates):
    # visualize in 3d
    x = [i[0] for i in coordinates]
    y = [i[1] for i in coordinates]
    z = [i[2] for i in coordinates]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Visualization')

    plt.show()


# important: https://github.com/windowsub0406/KITTI_Tutorial/blob/master/display_groundtruth.ipynb




def load_labels(label_path):


    '''

    function that loads labels from file
    designed for the KITTI velodyne: 
    args(str): path to labels
    for better understanding of the kitti point cloud labels: https://medium.com/@abdulhaq.ah/explain-label-file-of-kitti-dataset-738528de36f4

    '''


    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            cls = parts[0]
            truncation = float(parts[1]) 
            occlusion = int(parts[2])
            alpha = float(parts[3])
            bbox_2d = list(map(float, parts[4:8])) 
            dimensions = list(map(float, parts[8:11])) 
            location = list(map(float, parts[11:14]))  
            rotation_y = float(parts[14])
            labels.append({
                'class': cls,
                'truncation': truncation,
                'occlusion': occlusion,
                'alpha': alpha,
                'bbox_2d': bbox_2d,
                'dimensions': dimensions,
                'location': location,
                'rotation_y': rotation_y
            })
    return labels




# for details: https://stackoverflow.com/questions/62938546/how-to-draw-bounding-boxes-and-update-them-real-time-in-python

def box_center_to_corner(box_center):

    '''
    acts like a helper function
    box_center(list): takes an array containing [x,y,z,h,w,l,r], 
    output: an [8, 3] matrix that represents the [x, y, z] for each 8 corners of the box

    '''
    corner_boxes = np.zeros((8, 3))

    translation = box[0:3]
    h, w, l = size[3], size[4], size[5]
    rotation = box[6]

    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    eight_points = np.tile(translation, (8, 1))


    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()



lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        [4, 5], [5, 6], [6, 7], [4, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]


colors = [[1, 0, 0] for _ in range(len(lines))]


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(corner_box)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

# Create a visualization object and window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Display the bounding boxes:
vis.add_geometry(corner_box)






