import os
import pandas as pd

def label_data(path, label, out_folder):
    """
    Labels all frames in each CSV file in the specified folder with the given label.

    Params:
    path (str): Path to the folder containing the CSV files.
    label (str or int): The label to assign to all frames in each file.

    Returns:
    dict: A dictionary where keys are filenames and values are DataFrames with labeled data.
    """
    labeled_data = {}

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path)
            df['label'] = label
            labeled_data[filename] = df

            # Save the labeled DataFrame as an HDF5 file
            hdf5_file_path = os.path.join(out_folder, filename.replace('.csv', '.h5'))
            df.to_hdf(hdf5_file_path, key='df', mode='w')

    return labeled_data
