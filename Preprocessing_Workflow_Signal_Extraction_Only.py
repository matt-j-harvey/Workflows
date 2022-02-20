import sys
from PyQt5.QtWidgets import QApplication
import time
import shutil
import os
import datetime

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing/Signal_Extraction")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing/Brain_Registration")

import Get_Max_Projection
import Align_Skeleton_Integrated
import Align_Widefield_Sessions_Integrated
import Combined_Heamocorrection_Registration


# Define Global Variables - Look I Have to Have Them To Enable Sequential Windows In QT, This Was The Least Inelegant Solution
global skeleton_window
global matching_window
global base_directory


def extract_signal(base_directory, template_directory):

    # Copy Mask From Template To Base Directory
    source = os.path.join(template_directory, "mask.npy")
    destination = os.path.join(base_directory, "mask.npy")
    shutil.copy(source, destination)

    Combined_Heamocorrection_Registration.perform_heamocorrection(base_directory)



template_directory = "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"


session_list = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",
                "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_08_Transition_Imaging"]

for base_directory in session_list:

    # Print Start Time
    print("Processing: ", base_directory)
    print("Starting at: ", datetime.datetime.now())
    extract_signal(base_directory, template_directory)
    print("Finished at: ", datetime.datetime.now())