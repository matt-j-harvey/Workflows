import os
import sys
import numpy as np

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")

import Downsample_Running_Trace
import Create_Activity_Tensor
import Create_Video_From_Tensor
import Quantify_Region_Responses



def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)





def uncorrected_workflow(base_directory, onsets_file_list, tensor_names, start_window, stop_window, experiment_name, plot_titles, recalculate=False):


    # Check Workflow Directories
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, experiment_name)
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, experiment_name)

    check_directory(video_directory)
    check_directory(video_save_directory)
    check_directory(plot_directory)
    check_directory(current_plot_directory)

    # Get Activity Tensors
    number_of_conditions = len(onsets_file_list)
    for condition_index in range(number_of_conditions):
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_names[condition_index], running_correction=False)


    # View Individual Movie
    if len(onsets_file_list) == 2:
        Create_Video_From_Tensor.create_single_mouse_comparison_video(base_directory, tensor_names, start_window, stop_window, plot_titles, video_save_directory)

    elif len(onsets_file_list) == 3:
        Create_Video_From_Tensor.create_single_mouse_comparison_video_3_conditions(base_directory, tensor_names, start_window, stop_window, plot_titles, video_save_directory)

    # Plot Region Responses
    """
    condition_names = [tensor_names[0], tensor_names[1]]
    condition_1_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Corrected_Tensor.npy"))
    condition_2_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Corrected_Tensor.npy"))
    activity_tensor_list = [condition_1_tensor, condition_2_tensor]
    Quantify_Region_Responses.get_region_response_single_mouse(base_directory, activity_tensor_list, start_window, stop_window, condition_names, current_plot_directory, baseline_normalise=False)
    """


#"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


"""
start_window = -10
stop_window = 280
onset_files = [["perfect_transition_onsets.npy"], ["odour_expected_present_onsets.npy"], ["odour_not_expected_not_present_onsets.npy"]]
tensor_names = ["perfect_transitions", "odour_expected_present", "odour_not_expected_not_present"]
experiment_name = "Absence_Of_Expected_Odour"
plot_titles = ["perfect_transitions", "odour_expected_present", "odour_not_expected_not_present"]
"""


start_window = -40
stop_window = 40
onset_files = [["odour_2_cued_onsets.npy"], ["odour_2_not_cued_onsets.npy"]]
tensor_names = ["Odour_2_Cued", "Odour_2_Not_Cued"]
experiment_name = "Odour_2_Cuing_Effect"
plot_titles = ["Odour_2_Cued", "Odour_2_Not_Cued"]

all_mice = controls + mutants
for base_directory in all_mice:
    uncorrected_workflow(base_directory, onset_files, tensor_names, start_window, stop_window, experiment_name, plot_titles)


start_window = -40
stop_window = 40
onset_files = [["odour_1_cued_onsets.npy"], ["odour_1_not_cued_onsets.npy"]]
tensor_names = ["Odour_1_Cued", "Odour_1_Not_Cued"]
experiment_name = "Odour_1_Cuing_Effect"
plot_titles = ["Odour_1_Cued", "Odour_1_Not_Cued"]

all_mice = controls + mutants
for base_directory in all_mice:
    uncorrected_workflow(base_directory, onset_files, tensor_names, start_window, stop_window, experiment_name, plot_titles)
