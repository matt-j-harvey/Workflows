import tables
import h5py
import sys
import numpy as np
import os
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")

import Widefield_General_Functions



def convert_tables_to_hdf5(base_directory):

    # Extract Widefield Data
    widefield_file = base_directory + "/Delta_F.h5"
    widefield_data_file = tables.open_file(widefield_file, mode='r')
    widefield_data = widefield_data_file.root['Data']

    # Get Chunk Strucrure
    number_of_frames = np.shape(widefield_data)[0]
    number_of_pixels = np.shape(widefield_data)[1]
    print("Number Of Frames", number_of_frames)
    print("Nubmer of Pixels", number_of_pixels)

    preferred_chunk_size = 20000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Widefield_General_Functions.get_chunk_structure(preferred_chunk_size, number_of_frames)

    print("Reformatting: ", base_directory)

    output_file = os.path.join(base_directory, "Delta_F_Registered.hdf5")

    with h5py.File(output_file, "w") as f:
        dataset = f.create_dataset("Data", (number_of_frames, number_of_pixels), dtype=np.float32, chunks=True, compression="gzip")

        for chunk_index in range(number_of_chunks):
            print("Chunk:", str(chunk_index).zfill(2), "of", number_of_chunks)
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])
            dataset[+chunk_start:chunk_stop] = widefield_data[chunk_start:chunk_stop]




"""
"/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NRXN78.1A/2020_12_09_Switching",
"/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK4.1B/2021_03_04_Switching_Imaging",
"/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK7.1B/2021_03_02_Switching_Imaging",
"""

file_list = ["/media/matthew/Seagate Expansion Drive1/Switching_Analysis/Wildtype/NXAK14.1A/2021_06_09_Switching_Imaging"]


for base_directory in file_list:
    convert_tables_to_hdf5(base_directory)
