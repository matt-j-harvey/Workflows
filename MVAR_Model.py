import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def extract_running_tensor(downsampled_running_trace, onset_group, start_window, stop_window):

    running_tensor = []

    # Iterate Through Each Trial Onset
    for onset in onset_group:

        # Get Trial Start and Stop Times
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        # Extract Trial Data and Shifted Trial Data
        trial_data = downsampled_running_trace[trial_start:trial_stop]

        # Add To Running List
        running_tensor.append(trial_data)

    # Flatten Tensor
    number_of_trials = len(onset_group)
    trial_length = stop_window - start_window
    running_tensor = np.reshape(running_tensor, (number_of_trials * trial_length, 1))

    return running_tensor


def load_activity_tensors(base_directory, onset_list, start_window, stop_window):

    # Create Empty Lists To Hold Data
    activity_tensor = []
    preceeding_activity_tensor = []

    # Load Atlas Delta F
    atlas_delta_f = np.load(os.path.join(base_directory, "Allen_Region_Delta_F.npy"))

    # Iterate Through Each Trial Onset
    for onset in onset_list:

        # Get Trial Start and Stop Times
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        # Extract Trial Data and Shifted Trial Data
        trial_data = atlas_delta_f[trial_start:trial_stop]
        preceeding_trial_data = atlas_delta_f[trial_start-1:trial_stop-1]

        """
        trial_baseline = atlas_delta_f[trial_start-10:trial_start]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial_data = np.subtract(trial_data, trial_baseline)
        preceeding_trial_data = np.subtract(preceeding_trial_data, trial_baseline)
        """

        # Add These To Respective Lists
        activity_tensor.append(trial_data)
        preceeding_activity_tensor.append(preceeding_trial_data)

    # Convert Lists To Arrays
    activity_tensor = np.array(activity_tensor)
    preceeding_activity_tensor = np.array(preceeding_activity_tensor)

    # Flatten These
    number_of_trials = len(onset_list)
    trial_length = stop_window - start_window
    number_of_regions = np.shape(activity_tensor)[2]

    activity_tensor = np.reshape(activity_tensor, (number_of_trials * trial_length, number_of_regions))
    preceeding_activity_tensor = np.reshape(preceeding_activity_tensor, (number_of_trials * trial_length, number_of_regions))


    # Exclude Dubious Regions
    print("Activity Tensor Shape", np.shape(activity_tensor))
    print("Preceeding Activity Tensor Shape", np.shape(preceeding_activity_tensor))

    # Load Pixel assignments
    pixel_assigments = list(np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy"))
    unique_labels = list(set(pixel_assigments))
    print("unique labels", unique_labels)

    excluded_label_list = [0, 1, 3, 4, 5, 10, 18, 17]

    for label in excluded_label_list:
        lable_index = unique_labels.index(label)
        activity_tensor[:, lable_index] = 0
        preceeding_activity_tensor[:, lable_index] = 0


    return activity_tensor, preceeding_activity_tensor



def create_stimuli_regressors(stimuli_trials, trial_length, stimuli_regressor_matrix, start_index, stimuli_index):

    trial_start = start_index
    stimuli_start = stimuli_index * trial_length

    for trial_index in range(stimuli_trials):
        trial_stop = trial_start + trial_length
        stimuli_stop = stimuli_start + trial_length

        stimuli_regressor_matrix[trial_start:trial_stop, stimuli_start:stimuli_stop] = np.identity(trial_length)

        trial_start += trial_length

    return stimuli_regressor_matrix



def plot_raster(raster, trial_length):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(np.transpose(raster), cmap='jet', vmin=0, vmax=1)
    forceAspect(axis_1, aspect=2)

    number_of_trials = np.shape(raster)[0]
    for x in range(0, number_of_trials, trial_length):
        axis.axvline(x)

    plt.show()


def plot_raster_difference(raster, trial_length):

    raster_magnitude = np.max(np.abs(raster))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(np.transpose(raster), cmap='bwr', vmin=-1 * raster_magnitude, vmax=1 * raster_magnitude)
    forceAspect(axis_1, aspect=2)

    number_of_trials = np.shape(raster)[0]
    for x in range(0, number_of_trials, trial_length):
        axis.axvline(x)

    plt.show()



def fit_mvar_model(base_directory, onset_file_list, start_window, stop_window):


    onset_group_list = []

    # Load Onsets
    for onset_file in onset_file_list:
        onset_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))
        onset_group_list.append(onset_list)


    # Load Activity Tensors
    activity_tensor_list = []
    preceeding_activity_tensor_list = []
    for onset_group in onset_group_list:
        activity_tensor, preceeding_activity_tensor = load_activity_tensors(base_directory, onset_group, start_window, stop_window)
        activity_tensor_list.append(activity_tensor)
        preceeding_activity_tensor_list.append(preceeding_activity_tensor)


    activity_tensor_list = np.vstack(activity_tensor_list)
    preceeding_activity_tensor_list = np.vstack(preceeding_activity_tensor_list)

    """
    print("Activity Tensor List Shape", np.shape(activity_tensor_list))
    activity_tensor_list = stats.zscore(activity_tensor_list, axis=0)
    preceeding_activity_tensor_list = stats.zscore(preceeding_activity_tensor_list, axis=0)
    activity_tensor_list = np.nan_to_num(activity_tensor_list)
    preceeding_activity_tensor_list = np.nan_to_num(preceding_activity_tensor_list)
    """

    delta_activity_tensor = np.subtract(activity_tensor_list, preceeding_activity_tensor_list)

    trial_length = stop_window - start_window
    plot_raster(activity_tensor_list, trial_length)
    plot_raster(preceeding_activity_tensor_list, trial_length)
    plot_raster_difference(delta_activity_tensor)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(np.transpose(delta_activity_tensor), cmap='bwr', vmin=-1, vmax=1)
    forceAspect(axis_1, aspect=2)
    plt.show()



    # Get Total Number Of Trials
    total_number_of_trials = 0
    number_of_stimuli = len(onset_group_list)
    trial_length = stop_window - start_window
    for stimuli_index in range(number_of_stimuli):
        stimuli_trials = len(onset_group_list[stimuli_index])
        total_number_of_trials += stimuli_trials


    # Create Stimuli Regressors
    stimuli_regressor_matrix = np.zeros((total_number_of_trials * trial_length, (number_of_stimuli * trial_length)))

    start_index = 0
    for stimuli_index in range(number_of_stimuli):
        stimuli_trials = len(onset_group_list[stimuli_index])
        stimuli_regressor_matrix = create_stimuli_regressors(stimuli_trials, trial_length, stimuli_regressor_matrix, start_index, stimuli_index)
        start_index += stimuli_trials * trial_length

    plt.imshow(stimuli_regressor_matrix)
    plt.show()

    # Get Downsampled Running Speed
    downsampled_running_trace = np.load(os.path.join(base_directory, "Movement_Controls", "Downsampled_Running_Trace.npy"))
    running_regressor_list = []
    for onset_group in onset_group_list:
        running_tensor = extract_running_tensor(downsampled_running_trace, onset_group, start_window, stop_window)
        running_regressor_list.append(running_tensor)
    running_regressor_list = np.vstack(running_regressor_list)


    # Transpose These Tensors
    delta_activity_tensor = np.transpose(delta_activity_tensor)
    preceeding_activity_tensor_list = np.transpose(preceeding_activity_tensor_list)
    stimuli_regressor_matrix = np.transpose(stimuli_regressor_matrix)
    running_regressor_list = np.transpose(running_regressor_list)


    design_matrix = np.vstack([preceeding_activity_tensor_list]) #stimuli_regressor_matrix, running_regressor_list

    # Delta Activity Tensor
    print("Delta Activity Tensor Shape", np.shape(delta_activity_tensor))

    # Preceeding Activity Tensor Shape
    print("Preceeding Activity Tensor Shape", np.shape(preceeding_activity_tensor_list))

    # Stimuli Regressor Tensor
    print("Stimuli Regressor Tensor", np.shape(stimuli_regressor_matrix))

    # Running Tensor
    print("Running Tensor", np.shape(running_regressor_list))

    print("Design Matrix Shaoe", np.shape(design_matrix))

    #model = Ridge()
    #model.fit(X=np.transpose(design_matrix), y=np.transpose(delta_activity_tensor))
    #coefficients = model.coef_

    coefficients, residuals, rank, singular_values  = np.linalg.lstsq(np.transpose(design_matrix), np.transpose(delta_activity_tensor))

    print("Coefficients", np.shape(coefficients))

    connectivity_matrix = coefficients[:, 0:45]



    plt.imshow(connectivity_matrix, cmap='bwr', vmax=np.max(np.abs(connectivity_matrix)), vmin=-1 * np.max(np.abs(connectivity_matrix)))
    plt.show()

    return connectivity_matrix



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

pixel_assignement_image = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")
plt.imshow(pixel_assignement_image)
plt.show()

start_window = -10
stop_window = 20
context_1_onset_files = ["visual_context_stable_vis_1_onsets.npy", "visual_context_stable_vis_2_onsets.npy"]
context_2_onset_files = ["odour_context_stable_vis_1_onsets.npy", "odour_context_stable_vis_2_onsets.npy"]

context_1_connectivity = fit_mvar_model(controls[-1], context_1_onset_files, start_window, stop_window)
context_2_connectivity = fit_mvar_model(controls[-1], context_2_onset_files, start_window, stop_window)

connectivity_matrix = np.subtract(context_1_connectivity, context_2_connectivity)

plt.imshow(connectivity_matrix, cmap='bwr', vmax=np.max(np.abs(connectivity_matrix)), vmin=-1 * np.max(np.abs(connectivity_matrix)))
plt.show()
