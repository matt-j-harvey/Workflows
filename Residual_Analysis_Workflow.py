import os
import sys
import numpy as np

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")


import Downsample_Running_Trace
import Create_Activity_Tensor
import Create_Video_From_Tensor
import Quantify_Region_Responses
import Widefield_General_Functions
import Regress_Out_Running_Related_Activity



def get_onsets(base_directory, onsets_file_list):

    onsets = []
    for onsets_file in onsets_file_list:
        onsets_file_contents = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file))
        for onset in onsets_file_contents:
            onsets.append(onset)

    return onsets


def get_onsets_list(base_directory, onsets_file_list):

    onsets_list = []
    for onset_group in onsets_file_list:
        onsets = get_onsets(base_directory, onset_group)
        onsets_list.append(onsets)

    return onsets_list



def residual_analysis_workflow_single_mouse(base_directory, onsets_file_list, tensor_names, start_window, stop_window, plot_titles, experiment_name):


    # Check Workflow Directories
    movement_controls_directory = os.path.join(base_directory, "Movement_Controls")
    regression_coefficients_directory = os.path.join(movement_controls_directory, "Running_Regression_Coefficients")
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, experiment_name)
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, tensor_names[0] + "_" + tensor_names[1] + "_Residuals")

    Widefield_General_Functions.check_directory(movement_controls_directory)
    Widefield_General_Functions.check_directory(regression_coefficients_directory)
    Widefield_General_Functions.check_directory(video_directory)
    Widefield_General_Functions.check_directory(video_save_directory)
    Widefield_General_Functions.check_directory(plot_directory)
    Widefield_General_Functions.check_directory(current_plot_directory)

    # Downsample Running Trace
    downsampled_running_trace_file = os.path.join(movement_controls_directory, "Downsampled_Running_Trace.npy")
    if not os.path.exists(downsampled_running_trace_file):
        print("Downsampling Running Traces...")
        Downsample_Running_Trace.downsample_running_trace(base_directory)
    else:
        print("Downsampled Running Traces Already Calculated")


    # Get Activity Tensors
    activity_tensor_list = []
    number_of_conditions = len(tensor_names)
    for condition_index in range(number_of_conditions):
        activity_tensor = Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_names[condition_index])
        activity_tensor_list.append(activity_tensor)

    # Load Onsets List
    onsets_list =get_onsets_list(base_directory, onsets_file_list)

    print("Activity Tensor List", len(activity_tensor_list))
    for tensor in activity_tensor_list:
        print(np.shape(tensor))
    print("Onsets List", len(onsets_list))
    for onsets in onsets_list:
        print(len(onsets))

    # Perform Running Regression Correction
    predicted_tensor_list, corrected_tensor_list = Regress_Out_Running_Related_Activity.regress_out_running(base_directory, activity_tensor_list, onsets_list, start_window, stop_window, experiment_name)


    # View Individual Movie
    if len(predicted_tensor_list) == 2:

        mean_activity_tensor_list = []
        for tensor in corrected_tensor_list:
            print("Tensor Shape", np.shape(tensor))
            mean_tensor = np.mean(tensor, axis=0)
            print("Mean Tensor Shape", np.shape(mean_tensor))
            mean_activity_tensor_list.append(mean_tensor)

        Create_Video_From_Tensor.create_generic_comparison_video(base_directory, mean_activity_tensor_list, start_window, stop_window, plot_titles, video_save_directory)

    """
    # Plot Region Responses
    condition_names = [tensor_names[0], tensor_names[1]]
    condition_1_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Corrected_Tensor.npy"))
    condition_2_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Corrected_Tensor.npy"))
    activity_tensor_list = [condition_1_tensor, condition_2_tensor]
    Quantify_Region_Responses.get_region_response_single_mouse(base_directory, activity_tensor_list, start_window, stop_window, condition_names, current_plot_directory, baseline_normalise=False)
    """

controls = [
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_08_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_31_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_15_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",

            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_10_29_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK22.1A/2021_11_05_Transition_Imaging"
            ]




experiment_name = "Vis 2 Contextual Modulation Running Residuals"
start_window = -77
stop_window = 56
onset_files = [["visual_context_stable_vis_2_onsets.npy"], ["odour_context_stable_vis_2_onsets.npy"]]
tensor_names = ["Visual_Context_Stable_Vis_2_Running_Residual", "Odour_Context_Stable_Vis_2_Running_Residual"]
plot_titles = ["Visual_Context_Stable_Vis_2", "Odour_Context_Stable_Vis_2"]
behavioural_traces = ["Running", "Lick", "Visual 2"]
combined_video_save_directory = r"/home/matthew/Documents/Thesis_Comitte_24_02_2022/Vis_2_Contextual_Modulation_Controls_Residuals"

# Get For Each Mouse
for base_directory in controls:
    residual_analysis_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, plot_titles, experiment_name)

# Get For Group
#uncorrected_workflow_group(controls, onset_files, tensor_names, plot_titles, start_window, stop_window, combined_video_save_directory, behavioural_traces)

