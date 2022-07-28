import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tables

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")

#import Downsample_Running_Trace
import Workflow_Utils
import Create_Activity_Tensor
import Create_Video_From_Tensor_2
import Quantify_Region_Responses
import Create_Behaviour_Tensor




def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)





def uncorrected_workflow_single_mouse(base_directory, onsets_file_list, tensor_names, start_window, stop_window, selected_behaviour_traces, experiment_name, plot_titles, difference_conditions=False, normalised=False):


    # Check Workflow Directories
    video_directory = os.path.join(base_directory, "Response_Videos")
    video_save_directory = os.path.join(video_directory, experiment_name)
    plot_directory = os.path.join(base_directory, "Plots")
    current_plot_directory = os.path.join(plot_directory, experiment_name)

    check_directory(video_directory)
    check_directory(video_save_directory)
    check_directory(plot_directory)
    check_directory(current_plot_directory)


    # Create Behaviour Tensor
    behaviour_tensor_dict_list = []
    for onsets_file in onsets_file_list:
        behaviour_tensor_dict = Create_Behaviour_Tensor.create_behaviour_tensor_downsampled_ai(base_directory, onsets_file, start_window, stop_window, selected_behaviour_traces)
        behaviour_tensor_dict_list.append(behaviour_tensor_dict)

    """
    # Create Activity Tensors
    number_of_conditions = len(onsets_file_list)
    for condition_index in range(number_of_conditions):
        Create_Activity_Tensor.create_activity_tensor(base_directory, onsets_file_list[condition_index], start_window, stop_window, tensor_names[condition_index])
    """

    # Load Activity Tensors
    mean_activity_tensor_list = []
    for tensor_name in tensor_names:
        print("Loading: ", tensor_name)
        tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_name + "_Activity_Tensor.npy"))
        mean_tensor = np.mean(tensor, axis=0)

        # Unnormalised
        if normalised == False:
            mean_tensor = unnormalise_tensor(base_directory, mean_tensor)

        mean_activity_tensor_list.append(mean_tensor)

    # View Individual Movie
    indicies, image_height, image_width = Workflow_Utils.load_generous_mask(base_directory)
    Create_Video_From_Tensor_2.create_activity_video(indicies, image_height, image_width, mean_activity_tensor_list, start_window, stop_window, plot_titles, video_save_directory, behaviour_tensor_dict_list, selected_behaviour_traces, difference_conditions=difference_conditions)

    # Plot Region Responses
    """
    condition_names = [tensor_names[0], tensor_names[1]]
    condition_1_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[0] + "_Corrected_Tensor.npy"))
    condition_2_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", tensor_names[1] + "_Corrected_Tensor.npy"))
    activity_tensor_list = [condition_1_tensor, condition_2_tensor]
    Quantify_Region_Responses.get_region_response_single_mouse(base_directory, activity_tensor_list, start_window, stop_window, condition_names, current_plot_directory, baseline_normalise=False)
    """

def align_activity_tensor(base_directory, activity_tensor):

    consensus_mask = np.load("/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/mask.npy")
    #plt.imshow(consensus_mask)
    #plt.show()

    # Load Consensus Aligned Mask
    consensus_indicies, consensus_image_height, consensus_image_width = Workflow_Utils.load_consensus_mask()

    # Load Mask
    indicies, image_height, image_width = Workflow_Utils.load_generous_mask(base_directory)

    # Load Alignment Dictionary
    alignment_dictionary = np.load(os.path.join(base_directory, "Cluster_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    aligned_images = []
    for frame in activity_tensor:

        # Reconstruct Image
        frame = Workflow_Utils.create_image_from_data(frame, indicies, image_height, image_width)

        # Align Image
        frame = Workflow_Utils.transform_image(frame, alignment_dictionary, invert=False)

        # Flatten Image
        frame =  np.ndarray.reshape(frame, (image_height * image_width))

        # Take Indicies
        frame = frame[consensus_indicies]

        # Add To List
        aligned_images.append(frame)

    # Convert To Array
    aligned_images = np.array(aligned_images)


    return aligned_images


def unnormalise_tensor(base_directory, tensor):

    # Load Meta-Data
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_file = tables.open_file(delta_f_file, mode='r')
    baseline_values = np.array(delta_f_file.root.pixel_baseline_list)
    maximum_values = np.array(delta_f_file.root.pixel_maximum_list)
    baseline_values = np.transpose(baseline_values)
    maximum_values = np.transpose(maximum_values)

    # Load Mask
    indicies, image_height, image_width = Workflow_Utils.load_generous_mask(base_directory)

    # Convert To FLoat
    tensor = np.ndarray.astype(tensor, float)

    # Divide By UINT16 Max
    tensor = np.divide(tensor, 65535)

    # Divide By Max
    tensor = np.divide(tensor, maximum_values)
    tensor = np.nan_to_num(tensor)

    # Add Baselines
    tensor = np.add(tensor, baseline_values)
    tensor = np.nan_to_num(tensor)

    return tensor


def uncorrected_workflow_group(base_directory_list, onset_file_list, tensor_names, plot_titles, start_window, stop_window, save_directory, selected_behaviour_traces, difference_conditions=False, normalised=True):

    # Check Save Directory
    check_directory(save_directory)

    # Get Number Of Conditions
    number_of_conditions = len(tensor_names)

    # Create An Empty List To Hold A Behavioural Dictionary For Each Condition
    behaviour_dict_list = []

    for condition_index in range(number_of_conditions):

        # Create Dictionary To Hold List Of Mean Traces
        mean_behaviour_trace_dict = {}
        for trace in selected_behaviour_traces:
            mean_behaviour_trace_dict[trace] = []

        # Get Mean For Each Session
        onsets_file = onset_file_list[condition_index]
        for base_directory in base_directory_list:
            behaviour_tensor_dict = Create_Behaviour_Tensor.create_behaviour_tensor_downsampled_ai(base_directory, onsets_file, start_window, stop_window, selected_behaviour_traces)

            # Get Mean
            for trace in selected_behaviour_traces:
                mean_behaviour_trace_dict[trace].append(np.mean(behaviour_tensor_dict[trace], axis=0))

        # Get Group Mean For Each Behavioural Trace
        """
        for trace in selected_behaviour_traces:
            mean_behaviour_trace_dict[trace] = np.mean(mean_behaviour_trace_dict[trace], axis=0)
        """
        behaviour_dict_list.append(mean_behaviour_trace_dict)

    print("Behaviour Dict List", behaviour_dict_list)

    # Create List To Hold Activity Tensors
    activity_tensor_list = []

    for condition_index in range(number_of_conditions):

        condition_name = tensor_names[condition_index]
        condition_tensor_list = []

        for base_directory in base_directory_list:

            # Load Activity Tensor
            activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", condition_name + "_Activity_Tensor.npy"))

            # Get Average
            mean_activity = np.mean(activity_tensor, axis=0)

            # Unnormalised
            if normalised == False:
                mean_activity = unnormalise_tensor(base_directory, mean_activity)

            # Reconstruct and Align
            mean_activity = align_activity_tensor(base_directory, mean_activity)

            # Add To List
            condition_tensor_list.append(mean_activity)

        # Get Group Mean
        condition_tensor_list = np.array(condition_tensor_list)
        condition_mean_tensor = np.mean(condition_tensor_list, axis=0)
        activity_tensor_list.append(condition_mean_tensor)


    # View Individual Movie
    indicies, image_height, image_width = Workflow_Utils.load_consensus_mask()
    Create_Video_From_Tensor_2.create_activity_video(indicies, image_height, image_width, activity_tensor_list, start_window, stop_window, plot_titles, video_save_directory, behaviour_dict_list, selected_behaviour_traces, difference_conditions=difference_conditions)




session_list = [
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_16_Mirror_Imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_18_Mirror_Imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_23_mirror_imaging",
    r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/2022_05_27_mirror_imaging",
]

"""
experiment_name = "All Vis 1 v All Vis 2"
start_window = -10
stop_window = 70
onset_files = [["visual_1_all_onsets.npy"], ["visual_2_all_onsets.npy"]]
tensor_names = ["ALl_Vis_1", "All_Vis_2"]
plot_titles = ["ALl_Vis_1", "All_Vis_2"]
behavioural_traces = ["Running", "Visual 1", "Visual 2", "Odour 1", "Odour 2"]
"""

experiment_name = "Absence Of Expected Visual"
start_window = -10
stop_window = 100
onset_files = ["Visual_Expected_Present_onsets.npy", "Visual_Expected_Absent_onsets.npy", "Visual_Not_Expected_Absent_onsets.npy"]
tensor_names = ["Visual_Expected_Present", "Visual_Expected_Absent", "Visual_Not_Expected_Absent"]
plot_titles = ["Visual_Expected_Present", "Visual_Expected_Absent", "Visual_Not_Expected_Absent"]
behavioural_traces = ["Running", "Visual 1", "Visual 2", "Odour 1", "Odour 2"]
difference_conditions = [0,1]


onset_files = ["Visual_Expected_Present_onsets.npy", "Visual_Expected_Absent_onsets.npy"]
tensor_names = ["Visual_Expected_Present", "Visual_Expected_Absent"]
plot_titles = ["Visual_Expected_Present", "Visual_Expected_Absent"]


for base_directory in session_list:
    video_save_directory = os.path.join(base_directory, experiment_name)
    uncorrected_workflow_single_mouse(base_directory, onset_files, tensor_names, start_window, stop_window, behavioural_traces, experiment_name, plot_titles, difference_conditions, normalised=False)

"""
video_save_directory = r"/media/matthew/External_Harddrive_1/Processed_Widefield_Data/Beverly/Combined_Results/Absence_of_Expected_Visual"
uncorrected_workflow_group(session_list, onset_files, tensor_names, plot_titles, start_window, stop_window, video_save_directory, behavioural_traces, difference_conditions, normalised=False)
"""