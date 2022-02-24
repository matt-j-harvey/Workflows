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



def align_skeleton():
    skeleton_window.base_directory = base_directory

    skeleton_window.load_max_projection()
    skeleton_window.show()

def match_sessions():
    matching_window.template_directory = template_directory
    matching_window.matching_directory = base_directory
    matching_window.select_template_session()
    matching_window.select_matching_session()
    matching_window.show()

def extract_signal():

    # Copy Mask From Template To Base Directory
    source = os.path.join(template_directory, "mask.npy")
    destination = os.path.join(base_directory, "mask.npy")
    shutil.copy(source, destination)

    Combined_Heamocorrection_Registration.perform_heamocorrection(base_directory)

def null_function():
    pass



template_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"


session_list = [r"/media/matthew/Seagate Expansion Drive/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
                r"/media/matthew/Seagate Expansion Drive/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
                r"/media/matthew/Seagate Expansion Drive/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging"]


base_directory = session_list[2]

print("Aliging: ", base_directory)

# Make Windows
app = QApplication(sys.argv)
skeleton_window = Align_Skeleton_Integrated.align_skeleton_main()
matching_window = Align_Widefield_Sessions_Integrated.align_widefield_sessions_main()
skeleton_window.hide()
matching_window.hide()

# Setup Order Of Functions
skeleton_window.subsequent_function = match_sessions
matching_window.subsequent_function = null_function

# Get Max Projection
Get_Max_Projection.check_max_projection(base_directory)

# Set Skeleton
align_skeleton()

app.exec()