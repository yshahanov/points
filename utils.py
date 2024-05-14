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



