import sys
import numpy as np
import os

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")

import Create_Activity_Tensor
import Create_Video_From_Tensor



controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants =  ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]

all_mice = controls + mutants






trial_start = 0
trial_stop = 100

perfect_transition_mean_list = []
missed_transition_mean_list = []

save_directory = r"/media/matthew/29D46574463D2856/Transition_Type_Video"
template_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging"

for base_directory in mutants:
    print("Base Diretory: ", base_directory)

    # Create Activity Tensors
    Create_Activity_Tensor.create_activity_tensor(base_directory, ["perfect_transition_onsets.npy"], trial_start, trial_stop, "Perfect_Transitions")
    Create_Activity_Tensor.create_activity_tensor(base_directory, ["missed_transition_onsets.npy"],  trial_start, trial_stop, "Missed_Transitions")

Create_Video_From_Tensor.create_group_comparison_video(template_directory, controls, ["Perfect_Transitions", "Missed_Transitions"], trial_start, trial_stop, ["Perfect_Transitions", "Missed_Transitions"], save_directory)

"""
    # Load Activity Tensors
    perfect_transition_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", "Perfect_Transitions_Activity_Tensor.npy"))
    missed_transition_tensor  = np.load(os.path.join(base_directory, "Activity_Tensors", "Missed_Transitions_Activity_Tensor.npy"))



    if np.shape(perfect_transition_tensor)[0] > 0:
        perfect_transition_mean = np.mean(perfect_transition_tensor, axis=0)
        perfect_transition_mean_list.append(perfect_transition_mean)

    if np.shape(missed_transition_tensor)[0] > 0:
        missed_transition_mean = np.mean(missed_transition_tensor, axis=0)
        missed_transition_mean_list.append(missed_transition_mean)

    print(np.shape(perfect_transition_tensor))
    print(np.shape(missed_transition_tensor))
    """