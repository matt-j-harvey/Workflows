import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from scipy import stats
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def load_onsets(onset_file_list):

    onset_label_dict = {}
    combined_onset_list = []

    number_of_stimuli = len(onset_file_list)

    for stimuli_index in range(number_of_stimuli):
        onset_file = onset_file_list[stimuli_index]
        full_directory = os.path.join(base_directory, "Stimuli_Onsets", onset_file)
        stimuli_onset_list = np.load(full_directory)

        for onset in stimuli_onset_list:
            combined_onset_list.append(onset)
            onset_label_dict[onset] = stimuli_index

    #combined_onset_list.sort()

    label_list = []
    for onset in combined_onset_list:
        trial_type = onset_label_dict[onset]
        label_list.append(trial_type)

    return combined_onset_list, label_list



def load_activity_tensor(base_directory, onset_list, start_window, stop_window):

    # Load Data
    delta_f_file = os.path.join(base_directory, "Allen_Region_Delta_F.npy")
    delta_f_matrix = np.load(delta_f_file)

    # Extract Tensor
    activity_tensor = []
    for onset in combined_onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window
        trial_data = delta_f_matrix[trial_start:trial_stop]
        activity_tensor.append(trial_data)
    activity_tensor = np.array(activity_tensor)



    print("Activity Tensoe Shape", np.shape(activity_tensor))
    return activity_tensor


def convert_activity_tensor_to_noise_tensor(activity_tensor, trial_label_list):

    # Get Condition Means
    number_of_trials = np.shape(activity_tensor)[0]
    unique_trial_types = list(set(trial_label_list))
    trial_mean_list = []

    #  Get Mean For Each  Trial Type
    for trial_type in unique_trial_types:
        trial_mean = []

        for trial_index in range(number_of_trials):
            if trial_label_list[trial_index] == trial_type:
                trial_mean.append(activity_tensor[trial_index])

        trial_mean = np.array(trial_mean)
        trial_mean = np.mean(trial_mean, axis=0)
        trial_mean_list.append(trial_mean)

    # Subtract Mean For Each Trial
    subtracted_tensor = []
    for trial_index in range(number_of_trials):

        trial_activiy = activity_tensor[trial_index]

        trial_type = trial_label_list[trial_index]

        condition_mean = trial_mean_list[trial_type]

        subtracted_trial_activity = np.subtract(trial_activiy, condition_mean)

        subtracted_tensor.append(subtracted_trial_activity)

    subtracted_tensor = np.array(subtracted_tensor)

    return subtracted_tensor



def view_tensor(tensor, title="untitield"):

    number_of_trials = np.shape(tensor)[0]
    trial_length = np.shape(tensor)[1]
    number_of_regions = np.shape(tensor)[2]
    tensor = np.reshape(tensor, (number_of_trials * trial_length, number_of_regions))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(tensor), cmap='jet')
    axis_1.set_title(title)
    forceAspect(axis_1)
    plt.show()


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def get_cofluctuations(noise_tensor, trial_label_list, start_window, stop_window):

    number_of_trials = np.shape(noise_tensor)[0]
    trial_length = np.shape(noise_tensor)[1]
    number_of_regions = np.shape(noise_tensor)[2]
    noise_tensor = np.reshape(noise_tensor, (number_of_trials * trial_length, number_of_regions))

    # Perform Z Scoreing
    noise_tensor = np.transpose(noise_tensor)
    #plt.plot(noise_tensor[0])
    noise_tensor = stats.zscore(noise_tensor)
    #plt.plot(noise_tensor[0], c='g')
    #plt.show()
    print("Post Z Score Shape", np.shape(noise_tensor))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(noise_tensor, cmap='bwr', vmin=-3, vmax=3)
    axis_1.set_title("Z Score Noise Correlations")
    forceAspect(axis_1)
    plt.show()

    print("Number of regions", number_of_regions)

    number_of_timepoints = trial_length * number_of_trials

    exclusion_list = [0, 1, 10, 5, 3, 4, 17, 18]

    cofluctuation_array = []
    for region_index_1 in range(number_of_regions):
        region_1_trace = noise_tensor[region_index_1]

        for region_index_2 in range(region_index_1, number_of_regions):
            region_2_trace = noise_tensor[region_index_2]

            cofluctuation_trace = np.multiply(region_1_trace, region_2_trace)
            cofluctuation_array.append(cofluctuation_trace)

            #if region_index_1 == region_index_2:
                #cofluctuation_trace = np.zeros(number_of_timepoints)
            #else:


    cofluctuation_array = np.array(cofluctuation_array)
    print("cofluctation array shape", np.shape(cofluctuation_array))

    cofluctuation_magnitude = np.percentile(np.abs(cofluctuation_array), 95)

    #cofluctuation_array = sort_matrix(cofluctuation_array)

    number_of_trials = len(trial_label_list)
    trial_length = stop_window - start_window
    colour_list = ['blue', 'blue', 'green', 'green']


    # Get Condition Averages
    visual_average_list = []
    for trial_index in range(number_of_trials):
        trial_start = trial_index * trial_length
        trial_stop = trial_start + trial_length
        trial_type = trial_label_list[trial_index]
        if trial_type == 1:
            visual_average_list.append(cofluctuation_array[:, trial_start:trial_stop])

    visual_average_list = np.hstack(visual_average_list)
    print("Visual average list", np.shape(visual_average_list))
    visual_average = np.mean(visual_average_list, axis=1)

    sorted_visual_average = np.copy(visual_average)
    sorted_visual_average = list(sorted_visual_average)
    sorted_visual_average.sort()

    visual_average = list(visual_average)
    index_list = []
    for sorted_value in sorted_visual_average:
        index_list.append(visual_average.index(sorted_value))

    print("Index list", index_list)


    sorted_cofluctuation_matrix = []
    for index in index_list:
        sorted_cofluctuation_matrix.append(cofluctuation_array[index])

    sorted_cofluctuation_matrix = np.array(sorted_cofluctuation_matrix)
    print("sorted matrix shape", np.shape(sorted_cofluctuation_matrix))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.imshow(cofluctuation_array, cmap='bwr', vmin=-1 * cofluctuation_magnitude, vmax=cofluctuation_magnitude)
    axis_1.set_title("Coflucations")
    forceAspect(axis_1)


    print("Cofluctation array shaoe", np.shape(cofluctuation_array))
    for trial_index in range(number_of_trials):
        trial_start = trial_index * trial_length
        trial_stop = trial_start + trial_length

        trial_type = trial_label_list[trial_index]
        trial_colour = colour_list[trial_type]

        #print("Trial start", trial_start, "Trial Stop", trial_stop, "Trial Type", trial_type, "Trial colour", trial_colour)
        axis_1.axvspan(trial_start, trial_stop, alpha=0.1, color=trial_colour, ymin=0.95, ymax=1)

    plt.show()



    return cofluctuation_array


"""
def decode_context(cofluctation_matrix, trial_labels):

    all_timepoint_trial_labels =
"""


base_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_10_Transition_Imaging"
base_directory = "/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK7.1B/2021_03_23_Transition_Imaging"
#base_directory = r"/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_02_Transition_Imaging"



onset_file_list =["visual_context_stable_vis_1_onsets.npy",
                  "visual_context_stable_vis_2_onsets.npy",
                  "odour_context_stable_vis_1_onsets.npy",
                  "odour_context_stable_vis_2_onsets.npy"]

"""
onset_file_list =[
                  "visual_context_stable_vis_2_onsets.npy",
                  "odour_context_stable_vis_2_onsets.npy"]
"""


start_window = -30
stop_window = -2


# Load Onsets
combined_onset_list, trial_label_list = load_onsets(onset_file_list)

print("Combined onss", combined_onset_list)
print("Trial albel lists", trial_label_list)

# Load Activity Tensors
activity_tensor = load_activity_tensor(base_directory, combined_onset_list, start_window, stop_window)

# Convert Activity Tensors To Noise Tensors
noise_tensor = convert_activity_tensor_to_noise_tensor(activity_tensor, trial_label_list)
view_tensor(noise_tensor, title="Noise Activity")

# Get Cofluctuation Matrix
cofluctuation_matrix = get_cofluctuations(noise_tensor, trial_label_list, start_window, stop_window)

# Decompose Cofluctuation Matrix



"""
model = PCA(n_components=3)

cofluctuation_matrix = np.transpose(cofluctuation_matrix)
transformed_data = model.fit_transform(cofluctuation_matrix)

figure_1 = plt.figure(figsize=(4,4))
axis_1 = figure_1.add_subplot(111, projection='3d')

number_of_trials = np.shape(activity_tensor)[0]
print("number of trials", number_of_trials)
trial_length = stop_window - start_window
colourmap = cm.get_cmap('jet')
number_of_stimuli = np.max(trial_label_list)

for trial_index in range(number_of_trials):
    trial_start = trial_index * trial_length
    trial_stop = trial_start + trial_length
    trial_label = trial_label_list[trial_index]

    if trial_label == 1: trial_label = 0
    elif trial_label == 3: trial_label = 2

    print(trial_label)
    trial_colour = colourmap(float(trial_label)/number_of_stimuli)
    print(trial_colour)
    axis_1.scatter(transformed_data[trial_start:trial_stop, 0], transformed_data[trial_start:trial_stop, 1], transformed_data[trial_start:trial_stop, 2], color=trial_colour, alpha=0.3)
plt.show()
"""