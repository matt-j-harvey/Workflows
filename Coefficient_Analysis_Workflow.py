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


def coefficient_analysis_workflow(base_directory, onsets_file_list, tensor_names, start_window, stop_window, recalculate=False):


    # Check Workflow Directories
    movement_controls_directory = os.path.join(base_directory, "Movement_Controls")
    regression_coefficients_directory = os.path.join(movement_controls_directory, "Running_Regression_Coefficients")
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, tensor_names[0] + "_" + tensor_names[1] + "_Residuals")
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, tensor_names[0] + "_" + tensor_names[1] + "_Residuals")

    check_directory(movement_controls_directory)
    check_directory(regression_coefficients_directory)
    check_directory(video_directory)
    check_directory(video_save_directory)
    check_directory(plot_directory)
    check_directory(current_plot_directory)


    # Downsample Running Trace
    if recalculate == True:
        Downsample_Running_Trace.downsample_running_trace(base_directory)
    elif recalculate == False:
        # See If Downsampled Running Trace Exists
        downsampled_running_trace_file = os.path.join(movement_controls_directory, "Downsampled_Running_Trace.npy")
        if not os.path.exists(downsampled_running_trace_file):
            print("Downsampling Running Traces...")
            Downsample_Running_Trace.downsample_running_trace(base_directory)
        else:
            print("Downsampled Running Traces Already Calculated")

    # Get Running Regression Coefficients
    if recalculate == True:
        get_running_regression_coefficients(base_directory, "All_Times_Running")
    elif recalculate == False:
        running_coefficients_file = os.path.join(regression_coefficients_directory, "All_Times_Running_Coefficient_Vector.npy")
        running_intercepts_file = os.path.join(regression_coefficients_directory, "All_Times_Running_Coefficient_Vector.npy")
        if not os.path.exists(running_coefficients_file) or not os.path.exists(running_intercepts_file):
            get_running_regression_coefficients(base_directory, "All_Times_Running")
        else:
            print("Running Regression Coefficients Already Calculated")


    # Get Corrected Activity Tensors
    if recalculate == True:
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[0], trial_start, trial_stop, tensor_names[0], running_correction=True)
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[1], trial_start, trial_stop, tensor_names[1], running_correction=True)

    else:
        tensor_1_filepath = os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Corrected_Tensor.npy")
        if not os.path.exists(tensor_1_filepath):
            Create_Activity_Tensor.create_activity_tensor(base_directory, [onsets_file_list[0]], start_window, stop_window, tensor_names[0], running_correction=True)
        else:
            print("Corrected Tensor 1 Already Calculated")

        tensor_2_filepath = os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Corrected_Tensor.npy")
        if not os.path.exists(tensor_2_filepath):
            Create_Activity_Tensor.create_activity_tensor(base_directory, [onsets_file_list[1]], start_window, stop_window, tensor_names[1], running_correction=True)
        else:
            print("Corrected Tensor 2 Already Calculated")

    # View Individual Movie
    Create_Video_From_Tensor.create_single_mouse_comparison_video_with_correction(base_directory,
                                                                                  tensor_names[0],
                                                                                  tensor_names[1],
                                                                                  start_window,
                                                                                  stop_window,
                                                                                  [tensor_names[0], tensor_names[1]],
                                                                                  video_save_directory)

    # Plot Region Responses
    condition_names = [tensor_names[0], tensor_names[1]]
    condition_1_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Corrected_Tensor.npy"))
    condition_2_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Corrected_Tensor.npy"))
    activity_tensor_list = [condition_1_tensor, condition_2_tensor]
    Quantify_Region_Responses.get_region_response_single_mouse(base_directory, activity_tensor_list, start_window, stop_window, condition_names, current_plot_directory, baseline_normalise=False)


controls = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK14.1A/2021_06_17_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_04_02_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1A/2020_12_09_Switching_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN78.1D/2020_11_29_Switching_Imaging"]

mutants = ["/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


all_mice = controls + mutants

start_window = -10
stop_window = 40
onset_files = ["visual_context_stable_vis_2_frame_onsets.npy", "odour_context_stable_vis_2_frame_onsets.npy"]
tensor_names = ["Vis_2_Stable_Visual", "Vis_2_Stable_Odour"]


for base_directory in all_mice:
    residual_analysis_workflow(base_directory, onset_files, tensor_names, start_window, stop_window)
