import os
import sys
import numpy as np

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")

import Downsample_Running_Trace
import Create_Activity_Tensor
import Create_Video_From_Tensor
import Quantify_Region_Responses
import Create_Behaviour_Tensor



def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)





def uncorrected_workflow_single_mouse(base_directory, onsets_file_list, tensor_names, start_window, stop_window, experiment_name, plot_titles, recalculate=False):


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
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_names[condition_index])


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


def uncorrected_workflow_group(base_directory_list, onset_file_list, tensor_names, plot_titles, start_window, stop_window, save_directory, selected_ai_traces):


    # Check Save Directory
    check_directory(save_directory)

    # Get Number Of Conditions
    number_of_conditions = len(tensor_names)

    # Create List To Hold Activity Tensors
    activity_tensor_list = []

    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index]
        condition_tensor_list = []

        print(condition_name)
        for base_directory in base_directory_list:
            print(base_directory)

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_name + "_Activity_Tensor.npy"))

            # Get Average
            mean_activity = np.mean(activity_tensor, axis=0)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)
        activity_tensor_list.append(condition_mean_tensor)


    # Create An Empty List To Hold A Behavioural Dictionary For Each Condition
    behaviour_dict_list = []

    for condition_index in range(number_of_conditions):

        # Create Dictionary To Hold List Of Mean Traces
        mean_behaviour_trace_dict = {}
        for trace in selected_ai_traces:
            mean_behaviour_trace_dict[trace] = []

        # Get Mean For Each Session
        onsets_file = onset_file_list[condition_index][0]
        for base_directory in base_directory_list:
            behaviour_tensor = Create_Behaviour_Tensor.create_behaviour_tensor(base_directory, onsets_file, start_window, stop_window, selected_ai_traces)

            # Get Mean
            for trace in selected_ai_traces:
                mean_behaviour_trace_dict[trace].append(np.mean(behaviour_tensor[trace], axis=0))

        # Get Group Mean For Each Behavioural Trace
        for trace in selected_ai_traces:
            mean_behaviour_trace_dict[trace] = np.mean(mean_behaviour_trace_dict[trace], axis=0)

        behaviour_dict_list.append(mean_behaviour_trace_dict)

    if len(activity_tensor_list) == 2 :
        Create_Video_From_Tensor.create_generic_comparison_video_behaviour(base_directory_list[0], activity_tensor_list, start_window, stop_window, plot_titles, save_directory, behaviour_dict_list)

    if len(activity_tensor_list) == 3:
        Create_Video_From_Tensor.create_generic_comparison_video_3_conditions_behaviour(base_directory_list[0], activity_tensor_list, start_window, stop_window, plot_titles, save_directory, behaviour_dict_list)








controls = [
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]


mutants = [
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK24.1C/2021_11_10_Transition_Imaging",

"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_22_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_24_Transition_Imaging",
"/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging",
]


experiment_name = "Absence Of Expected Odour"
start_window = -10
stop_window = 85
onset_files = [["odour_not_expected_not_present_onsets.npy"], ["odour_expected_absent_onsets.npy"], ["odour_expected_present_onsets.npy"]]
tensor_names = ["Odour_Absent_Not_Expected", "Odour_Absent_Expected", "Odour_Present_Expected"]
plot_titles = ["Odour_Absent_Not_Expected", "Odour_Absent_Expected", "Odour_Present_Expected"]
behavioural_traces = ["Running", "Visual 1", "Visual 2", "Odour 1", "Odour 2"]
combined_video_save_directory = r"/home/matthew/Documents/Thesis_Comitte_24_02_2022/Absence_of_Expected_Odour_Controls_Raw"



# Get For Each Mouse
"""
for base_directory in mutants:
    uncorrected_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, experiment_name, plot_titles)
"""
# Get For Group
uncorrected_workflow_group(mutants, onset_files, tensor_names, plot_titles, start_window, stop_window, combined_video_save_directory, behavioural_traces)

