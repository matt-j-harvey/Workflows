import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Movement_Controls/Residual_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Trial_Aligned_Analysis")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Analysis/Functional_Connectivity")
sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")


import Create_Activity_Tensor
import Get_Region_Mean_Tensor
import Perform_Psychophysical_Interaction_Analysis
import Widefield_General_Functions


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)



def jointly_sort_matricies(key_matrix, other_matricies):

    # Cluster Matrix
    Z = ward(pdist(key_matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Key Matrix
    sorted_key_matrix = key_matrix[:, new_order][new_order]

    # Sort Other Matricies
    sorted_matrix_list = []
    for matrix in other_matricies:
        sorted_matrix = matrix[:, new_order][new_order]
        sorted_matrix_list.append(sorted_matrix)

    return sorted_key_matrix, sorted_matrix_list


def load_atlas_labels(selected_region_names):

    # Atlas Labels File
    atlas_labels = "/home/matthew/Documents/Github_Code/Widefield_Preprocessing/Allen_Atlas_Templates/Atlas_Labels.csv"

    with open(atlas_labels, mode='r') as inp:
        reader = csv.reader(inp)
        atlas_dicionary = {rows[1]: rows[0] for rows in reader}

    # Get Labels Of Selected Regions
    selected_region_labels = []
    for region_name in selected_region_names:
        region_label = atlas_dicionary[region_name]
        selected_region_labels.append(int(region_label))

    return selected_region_labels



def regress_out_running_from_activity_tensor(activity_tensor_list, bodycam_tensor):

    # Get Trials For Each Stimuli
    stimuli_1_trials = np.shape(activity_tensor_list[0])[0]

    # Create Concatenated Activity Tensor
    concatenated_activity_tensor = np.vstack(activity_tensor_list)

    #get Data Structure
    number_of_trials = np.shape(concatenated_activity_tensor)[0]
    trial_length = np.shape(concatenated_activity_tensor)[1]
    number_of_rois = np.shape(concatenated_activity_tensor)[2]
    number_of_components = np.shape(bodycam_tensor)[2]

    # Flatten Tensors
    flat_activity_tensor = np.reshape(concatenated_activity_tensor, (number_of_trials * trial_length, number_of_rois))
    flat_bodycam_tensor = np.reshape(bodycam_tensor, (number_of_trials * trial_length, number_of_components))

    # Create Model
    model = LinearRegression()
    model.fit(X=flat_bodycam_tensor, y=flat_activity_tensor)

    # Predict Activity from Bodycam
    predicited_activity = model.predict(X=flat_bodycam_tensor)

    # Correct Activity
    corrected_activity = np.subtract(flat_activity_tensor, predicited_activity)
    corrected_activity = np.ndarray.reshape(corrected_activity, (number_of_trials, trial_length, number_of_rois))

    # Split Back Into Stimuli Tensors
    stimuli_1_activity_tensor = corrected_activity[0:stimuli_1_trials]
    stimuli_2_activity_tensor = corrected_activity[stimuli_1_trials:]

    stimuli_1_bodycam_tensor = bodycam_tensor[0:stimuli_1_trials]
    stimuli_2_bodycam_tensor = bodycam_tensor[stimuli_1_trials:]

    return [stimuli_1_activity_tensor, stimuli_2_activity_tensor], [stimuli_1_bodycam_tensor, stimuli_2_bodycam_tensor]





def view_coefficient_vector(pixel_assignments, cluster_label_list, cluster_vector, indicies, image_height, image_width):

    # Create Blank Template
    template = np.zeros(np.shape(indicies))
    print("Template Shape", np.shape(template))

    # Fill Each Cluster
    number_of_clusters = len(cluster_vector)
    for cluster_index in range(number_of_clusters):

        cluster_value = cluster_vector[cluster_index]
        cluster_label = cluster_label_list[cluster_index]

        print("Cluster Index", cluster_index)
        print("Cluster Label", cluster_label)
        print("Cluster Value", cluster_value)

        # Fill Template
        pixel_indexes = np.where(pixel_assignments == cluster_label, 1, 0)
        pixel_indexes = list(np.nonzero(pixel_indexes))
        pixel_indexes.sort()
        template[pixel_indexes] = cluster_value


    # Create Image
    cluster_image = Widefield_General_Functions.create_image_from_data(template, indicies, image_height, image_width)

    return cluster_image


def view_ppi_modulation(base_directory, region_group_name):
    # Laod Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Pixel Assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))

    # Load PPI Coefs
    ppi_coef_file = os.path.join(base_directory, "Functional Connectivity Analysis", "Psychophysical Interactions", region_group_name + "_PPI_Coefs.npy")
    ppi_coefs = np.load(ppi_coef_file)

    plt.imshow(ppi_coefs)
    plt.show()

    context_1_coefs = ppi_coefs[:, 1]
    context_2_coefs = ppi_coefs[:, 2]
    modulation_coefs = np.subtract(context_1_coefs, context_2_coefs)

    context_1_image = view_coefficient_vector(pixel_assignments, cluster_list, context_1_coefs, indicies, image_height, image_width)
    context_2_image = view_coefficient_vector(pixel_assignments, cluster_list, context_2_coefs, indicies, image_height, image_width)
    modulation_image = view_coefficient_vector(pixel_assignments, cluster_list, modulation_coefs, indicies, image_height, image_width)

    context_1_image_magnitude = np.max(np.abs(context_1_coefs))
    context_2_image_magnitude = np.max(np.abs(context_2_coefs))
    modulation_magnitude = np.max(np.abs(modulation_coefs))

    rows = 1
    columns = 3
    figure_1 = plt.figure()
    context_1_axis = figure_1.add_subplot(rows, columns, 1)
    context_2_axis= figure_1.add_subplot(rows, columns, 2)
    modulation_axis = figure_1.add_subplot(rows, columns, 3)

    context_1_axis.imshow(context_1_image, cmap='bwr', vmin= -1 * context_1_image_magnitude, vmax=context_1_image_magnitude)
    context_2_axis.imshow(context_2_image, cmap='bwr', vmin=-1 * context_2_image_magnitude, vmax=context_2_image_magnitude)
    modulation_axis.imshow(modulation_image, cmap='bwr', vmin=-1 * modulation_magnitude, vmax=modulation_magnitude)
    plt.title(str(base_directory))
    plt.show()


def view_ppi_coefs(base_directory, region_group_name, start_window, stop_window):

    # Laod Mask
    indicies, image_height, image_width = Widefield_General_Functions.load_mask(base_directory)

    # Load Pixel Assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))

    print("Cluster List", cluster_list)

    """
    cluster_index_list = []
    for cluster in cluster_list:
        cluster_index = cluster_list.index(cluster)
        cluster_index_list.append(cluster_index)
    """


    # Load PPI Coefs
    ppi_coef_file = os.path.join(base_directory, "Functional Connectivity Analysis", "Psychophysical Interactions", region_group_name + "_PPI_Coefs.npy")
    ppi_coefs = np.load(ppi_coef_file)

    plt.imshow(ppi_coefs)
    plt.show()

    print("PPI Coef Shape", np.shape(ppi_coefs))

    # Get Window Structure
    start_index = 3
    number_of_conditions = 4
    window_size = stop_window - start_window

    cluster_magntiude = np.max(np.abs(ppi_coefs))

    plt.ion()
    for condition_index in range(start_index, start_index + (number_of_conditions * window_size), window_size):
        for timepoint_index in range(window_size):

            coef_index = condition_index + timepoint_index
            coefs = ppi_coefs[:, coef_index]

            cluster_image = view_coefficient_vector(pixel_assignments, cluster_list, coefs, indicies, image_height, image_width)


            plt.imshow(cluster_image, cmap='bwr', vmin=-1 * cluster_magntiude, vmax= cluster_magntiude)
            plt.title(str(timepoint_index))
            plt.draw()
            plt.pause(0.1)
            plt.clf()



def view_stimuli_regressors(base_directory, number_of_stimuli, start_window, stop_window):

    # Get Stimuli Details
    trial_length = stop_window - start_window

    # View Regressors
    indicies, height, width = Widefield_General_Functions.load_mask(base_directory)

    # Load Pixel Assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))

    # Load Meta List
    ppi_coef_meta_list = np.load(os.path.join(base_directory, "Functional Connectivity Analysis/Psychophysical Interactions/PPI_Coefs_All_Regions.npy"))
    print(np.shape(ppi_coef_meta_list))

    # Get Number Of Clusters
    number_of_clusters = np.shape(ppi_coef_meta_list)[0]

    # Get Trace Activity
    for stimuli_index in range(number_of_stimuli):
        cluster_trace_activity_list = []

        start = 3 + stimuli_index * trial_length
        stop = start + trial_length
        for cluster_index in range(number_of_clusters):
            cluster_stimuli_regressor = ppi_coef_meta_list[cluster_index, :, start:stop]
            cluster_activity_trace = np.mean(cluster_stimuli_regressor, axis=0)
            cluster_trace_activity_list.append(cluster_activity_trace)

        cluster_trace_activity_list = np.array(cluster_trace_activity_list)
        print("Cluster Delta F Trace", np.shape(cluster_trace_activity_list))

        activity_min = 0#np.min(cluster_trace_activity_list)
        activity_max = np.max(cluster_trace_activity_list)


        plt.ion()
        for timepoint_index in range(trial_length):
            timepoint_activity = cluster_trace_activity_list[:, timepoint_index]
            timepoint_activity_image = view_coefficient_vector(pixel_assignments, cluster_list, timepoint_activity, indicies, height, width)

            plt.title("Stimuli: " + str(stimuli_index) + " Timepoint: " + str(timepoint_index))
            plt.imshow(timepoint_activity_image, cmap='inferno', vmin=activity_min, vmax=activity_max)
            plt.draw()
            plt.pause(0.1)
            plt.clf()



def draw_brain_network(base_directory, adjacency_matrix, session_name):

    # Load Cluster Centroids
    cluster_centroids = np.load(base_directory + "/Cluster_Centroids.npy")

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)

    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(weights))

    # Get Edge Colours
    colourmap = cm.get_cmap('plasma')
    colours = []
    for weight in weights:
        colour = colourmap(weight)
        colours.append(colour)

    # Load Cluster Outlines
    cluster_outlines = np.load("/media/matthew/Seagate Expansion Drive2/Widefield_Imaging/Transition_Analysis/clean_clusters_outline.npy")
    plt.imshow(cluster_outlines)

    image_height = np.shape(cluster_outlines)[0]

    # Draw Graph
    # Invert Cluster Centroids
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])

    plt.title(session_name)
    nx.draw(graph, pos=inverted_centroids, node_size=1,  width=weights, edge_color=colours)
    plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    plt.close()





def view_modulation_matricies(session_list):


    context_1_matrix_list = []
    context_2_matrix_list = []
    modulation_matrix_list = []

    for base_directory in session_list:
        ppi_coef_meta_list = np.load(os.path.join(base_directory,   "Functional Connectivity Analysis/Psychophysical Interactions/PPI_Coefs_All_Regions.npy"))
        print(np.shape(ppi_coef_meta_list))

        context_1_matrix = ppi_coef_meta_list[:, :, 1]
        context_2_matrix = ppi_coef_meta_list[:, :, 2]

        # symmetrize
        #context_1_matrix = np.mean(np.array([context_1_matrix, np.transpose(context_1_matrix)]), axis=0)
        #context_2_matrix = np.mean(np.array([context_2_matrix, np.transpose(context_2_matrix)]), axis=0)


        difference_matrix = np.subtract(context_1_matrix, context_2_matrix)


        context_1_matrix_list.append(context_1_matrix)
        context_2_matrix_list.append(context_2_matrix)
        modulation_matrix_list.append(difference_matrix)

        context_1_matrix_magnitude = np.max(np.abs(context_1_matrix))
        context_2_matrix_magnitude = np.max(np.abs(context_2_matrix))
        difference_matrix_magnitude = np.max(np.abs(difference_matrix))

        #difference_matrix, [context_1_matrix, context_2_matrix] = jointly_sort_matricies(difference_matrix, [context_1_matrix, context_2_matrix])


        rows = 1
        columns = 3
        figure_1 = plt.figure()
        context_1_axis = figure_1.add_subplot(rows, columns, 1)
        context_2_axis = figure_1.add_subplot(rows, columns, 2)
        differnece_axis = figure_1.add_subplot(rows, columns, 3)
    
        print("context 2 mstrix shape", np.shape(context_1_matrix))
        figure_1.suptitle(str(base_directory.split('/')[-2:]))
        context_1_axis.imshow(context_1_matrix,     cmap='bwr', vmin=-1*context_1_matrix_magnitude,  vmax=context_1_matrix_magnitude)
        context_2_axis.imshow(context_2_matrix,     cmap='bwr', vmin=-1*context_2_matrix_magnitude,  vmax=context_2_matrix_magnitude)
        differnece_axis.imshow(difference_matrix,   cmap='bwr', vmin=-1*difference_matrix_magnitude, vmax=difference_matrix_magnitude)
        plt.show()



    modulation_matrix_list = np.array(modulation_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matrix_list, axis=0)

    plt.title("Unthresholded PPI Connectivity")
    plt.imshow(mean_modulation_matrix, cmap='bwr', vmin=-1*np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()

    #anyzmean_modulation_matrix = np.add(mean_modulation_matrix, np.transpose(mean_modulation_matrix))
    t_stats, p_values = stats.ttest_rel(context_1_matrix_list, context_2_matrix_list)

    p_threshold = 0.05
    thresholded_modulation_matrix = np.where(p_values < p_threshold, mean_modulation_matrix, 0)

    plt.title("Thresholded PPI Connectivity")
    plt.imshow(thresholded_modulation_matrix, cmap='bwr', vmin=-1*np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()


def get_significant_modulation(all_mice):


    region_list = ["Prinary_Visual", "Retrosplenial", "Secondary_Motor", "Somatosensory"]

    # Load
    condition_1_maps = []
    condition_2_maps = []

    # Create Comparison Images
    indicies, height, width = Widefield_General_Functions.load_mask(all_mice[0])

    for region in region_list:
        for base_directory in all_mice:

            ppi_coefs = np.load(os.path.join(base_directory, "Functional Connectivity Analysis", "Psychophysical Interactions", region + "_PPI_Coefs.npy"))

            condition_1_map = ppi_coefs[:, 1]
            condition_2_map = ppi_coefs[:, 2]
            modulation_map = np.subtract(condition_1_map, condition_2_map)

            context_1_magnitude = np.max(np.abs(condition_1_map))
            context_2_magnitude = np.max(np.abs(condition_2_map))
            modulation_magnitude = np.max(np.abs(modulation_map))

            context_1_image = Widefield_General_Functions.create_image_from_data(condition_1_map, indicies, height, width)
            context_2_image = Widefield_General_Functions.create_image_from_data(condition_2_map, indicies, height, width)
            modulation_image = Widefield_General_Functions.create_image_from_data(modulation_map, indicies, height, width)

            figure_1 = plt.figure()
            axis_1 = figure_1.add_subplot(1,3,1)
            axis_2 = figure_1.add_subplot(1,3,2)
            axis_3 = figure_1.add_subplot(1,3,3)

            axis_1.imshow(context_1_image, cmap='bwr', vmax=context_1_magnitude, vmin=-1 * context_1_magnitude)
            axis_2.imshow(context_2_image, cmap='bwr', vmax=context_2_magnitude, vmin=-1 * context_2_magnitude)
            axis_3.imshow(modulation_image, cmap='bwr', vmax=modulation_magnitude, vmin=-1 * modulation_magnitude)

            plt.show()

            condition_1_map_list.append(condition_1_map)
            condition_2_map_list.append(condition_2_map)

            condition_1_map_list = np.array(condition_1_map_list)
        condition_2_map_list = np.array(condition_2_map_list)
        t_stats, p_values = stats.ttest_rel(condition_1_map_list, condition_2_map_list, axis=0)
        image = Widefield_General_Functions.create_image_from_data(t_stats, indicies, height, width)
        image_magnitude = np.max(np.abs(image))

def flatten_region_tensor(tensor):
    number_of_trials = np.shape(tensor)[0]
    trial_length = np.shape(tensor)[1]
    tensor = np.reshape(tensor, (trial_length * number_of_trials))
    return tensor


def examine_motor_v1_relationship(base_directory, context_1_onsets, context_2_onsets, start_window, stop_window):

    # Load Pixel Assignments
    print(base_directory)
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))
    number_of_clusters = len(cluster_list)
    print("number of clusters", number_of_clusters)
    print("cluster list", cluster_list)

    # Show Pixel Assignmnent Image
    pixel_assignment_image = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    plt.imshow(pixel_assignment_image)
    plt.show()

    # Create Tensors
    number_of_context_1_conditions = len(context_1_onsets)
    number_of_context_2_conditions = len(context_2_onsets)

    context_1_activity_tensor_list = []
    for condition_index in range(number_of_context_1_conditions):
        activity_tensor = Create_Activity_Tensor.create_allen_atlas_activity_tensor(base_directory, context_1_onsets[condition_index], start_window, stop_window)
        activity_tensor = np.nan_to_num(activity_tensor)
        context_1_activity_tensor_list.append(activity_tensor)

    context_2_activity_tensor_list = []
    for condition_index in range(number_of_context_2_conditions):
        activity_tensor = Create_Activity_Tensor.create_allen_atlas_activity_tensor(base_directory, context_2_onsets[condition_index], start_window, stop_window)
        activity_tensor = np.nan_to_num(activity_tensor)
        context_2_activity_tensor_list.append(activity_tensor)


    v1_label = 45
    m2_label = 8

    v1_index = cluster_list.index(v1_label)
    m2_index = cluster_list.index(m2_label)

    v1_context_1_trace_list = []
    m2_context_1_trace_list = []
    v1_context_2_trace_list = []
    m2_context_2_trace_list = []

    # Create Region Tensors
    for condition_index in range(number_of_context_1_conditions):
        activity_tensor = context_1_activity_tensor_list[condition_index]
        v1_trace = activity_tensor[:, :, v1_index]
        m2_trace = activity_tensor[:, :, m2_index]

        # Subtract Means
        v1_mean = np.mean(v1_trace, axis=0)
        m2_mean = np.mean(m2_trace, axis=0)
        v1_trace = np.subtract(v1_trace, v1_mean)
        m2_trace = np.subtract(m2_trace, m2_mean)

        v1_trace = flatten_region_tensor(v1_trace)
        m2_trace = flatten_region_tensor(m2_trace)
        v1_context_1_trace_list.append(v1_trace)
        m2_context_1_trace_list.append(m2_trace)

    for condition_index in range(number_of_context_2_conditions):
        activity_tensor = context_2_activity_tensor_list[condition_index]
        v1_trace = activity_tensor[:, :, v1_index]
        m2_trace = activity_tensor[:, :, m2_index]

        # Subtract Means
        v1_mean = np.mean(v1_trace, axis=0)
        m2_mean = np.mean(m2_trace, axis=0)
        v1_trace = np.subtract(v1_trace, v1_mean)
        m2_trace = np.subtract(m2_trace, m2_mean)

        v1_trace = flatten_region_tensor(v1_trace)
        m2_trace = flatten_region_tensor(m2_trace)
        v1_context_2_trace_list.append(v1_trace)
        m2_context_2_trace_list.append(m2_trace)

    v1_context_1_trace = np.concatenate(v1_context_1_trace_list)
    m2_context_1_trace = np.concatenate(m2_context_1_trace_list)
    v1_context_2_trace = np.concatenate(v1_context_2_trace_list)
    m2_context_2_trace = np.concatenate(m2_context_2_trace_list)

    figure_1 = plt.figure()
    context_1_axis = figure_1.add_subplot(2, 1, 1)
    context_2_axis = figure_1.add_subplot(2, 1, 2)

    context_1_axis.plot(v1_context_1_trace, c='b')
    context_1_axis.plot(m2_context_1_trace, c='r')

    context_2_axis.plot(v1_context_2_trace, c='b')
    context_2_axis.plot(m2_context_2_trace, c='r')
    plt.show()

    figure_1 = plt.figure()
    context_1_axis = figure_1.add_subplot(2, 1, 1)
    context_2_axis = figure_1.add_subplot(2, 1, 2)

    context_1_axis.scatter(v1_context_1_trace, m2_context_1_trace, c='orange')
    context_2_axis.scatter(m2_context_2_trace, v1_context_2_trace, c='green')
    plt.show()




def psychophysical_interaction_workflow(base_directory, context_1_onsets, context_2_onsets, start_window, stop_window):

    # Load Pixel Assignments
    print(base_directory)
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))
    number_of_clusters = len(cluster_list)
    print("number of clusters", number_of_clusters)
    print("cluster list", cluster_list)

    # Show Pixel Assignmnent Image
    pixel_assignment_image = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    #plt.imshow(pixel_assignment_image)
    #plt.show()

    # Check Output Folders Exist
    functional_connectivity_directory = os.path.join(base_directory, "Functional Connectivity Analysis")
    ppi_save_directory = os.path.join(functional_connectivity_directory, "Psychophysical Interactions")
    beta_series_save_directory = os.path.join(functional_connectivity_directory, "Beta Series Correlations")
    check_directory(functional_connectivity_directory)
    check_directory(ppi_save_directory)

    # Create Tensors
    number_of_context_1_conditions = len(context_1_onsets)
    number_of_context_2_conditions = len(context_2_onsets)

    context_1_activity_tensor_list = []
    for condition_index in range(number_of_context_1_conditions):
        activity_tensor = Create_Activity_Tensor.create_allen_atlas_activity_tensor(base_directory, context_1_onsets[condition_index], start_window, stop_window)
        activity_tensor = np.nan_to_num(activity_tensor)
        context_1_activity_tensor_list.append(activity_tensor)

    context_2_activity_tensor_list = []
    for condition_index in range(number_of_context_2_conditions):
        activity_tensor = Create_Activity_Tensor.create_allen_atlas_activity_tensor(base_directory, context_2_onsets[condition_index], start_window, stop_window)
        activity_tensor = np.nan_to_num(activity_tensor)
        context_2_activity_tensor_list.append(activity_tensor)


    # Load Bodycam Tensors
    context_1_bodycam_tensor = np.load(os.path.join(beta_series_save_directory, "Condition_1_Bodycam_Tensor.npy"))
    context_2_bodycam_tensor = np.load(os.path.join(beta_series_save_directory, "Condition_2_Bodycam_Tensor.npy"))

    # Regress Out Movement
    #context_1_activity_tensor_list, context_1_bodycam_tensor_list = regress_out_running_from_activity_tensor(context_1_activity_tensor_list, condition_1_bodycam_tensor)
    #context_2_activity_tensor_list, context_2_bodycam_tensor_list = regress_out_running_from_activity_tensor(context_2_activity_tensor_list, condition_2_bodycam_tensor)


    ppi_coef_meta_list = []

    for region_label in cluster_list:

        allen_region_index = cluster_list.index(region_label)
        print("Region: ", region_label, "Index ", allen_region_index)
        #print("Region Index:", allen_region_index)

        # Create Region Tensors
        context_1_region_tensor_list = []
        for condition_index in range(number_of_context_1_conditions):
            activity_tensor = context_1_activity_tensor_list[condition_index]
            region_tensor = activity_tensor[:, :, allen_region_index]
            context_1_region_tensor_list.append(region_tensor)


        context_2_region_tensor_list = []
        for condition_index in range(number_of_context_2_conditions):
            activity_tensor = context_2_activity_tensor_list[condition_index]
            region_tensor = activity_tensor[:, :, allen_region_index]
            context_2_region_tensor_list.append(region_tensor)


        # Check Shape
        """
        for tensor in context_1_activity_tensor_list:
            print("Activity Tensor Shape", np.shape(tensor))

        for tensor in context_1_region_tensor_list:
            print("Region Tensor Shape", np.shape(tensor))
        """

        # Perform Noise Correlation Analysis
        ppi_coefs = Perform_Psychophysical_Interaction_Analysis.create_ppi_model(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list, context_1_bodycam_tensor, context_2_bodycam_tensor)
        ppi_coef_meta_list.append(ppi_coefs)

    ppi_coef_meta_list = np.array(ppi_coef_meta_list)
    print("Meta Array Shape", np.shape(ppi_coef_meta_list))

    ppi_coef_filename = "PPI_Coefs_All_Regions.npy"
    np.save(os.path.join(ppi_save_directory, ppi_coef_filename), ppi_coef_meta_list)






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


start_window = -10
stop_window = 20
context_1_onset_files = [["visual_context_stable_vis_1_onsets.npy"], ["visual_context_stable_vis_2_onsets.npy"]]
context_2_onset_files = [["odour_context_stable_vis_1_onsets.npy"],  ["odour_context_stable_vis_2_onsets.npy"]]

#context_1_onset_files = [["visual_context_stable_vis_2_onsets.npy"]]
#context_2_onset_files = [["odour_context_stable_vis_2_onsets.npy"]]

#context_1_onset_files = [["visual_context_stable_vis_1_onsets.npy"]]
#context_2_onset_files = [["odour_context_stable_vis_1_onsets.npy"]]



selected_regions = [['Primary_Visual_Left', 'Primary_Visual_Right'],
                    ['Retosplenlial_Dorsal'],
                    ['Secondary_Motor_Left','Secondary_Motor_Right'],
                    ['Primary_Somatosensory_Barrel_Left', 'Primary_Somatosensory_Barrel_Right']]

region_group_names = ["Prinary_Visual", "Retrosplenial", "Secondary_Motor", "Somatosensory"]


#all_mice = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging"]
#selected_regions, region_group_names

#view_modulation_matricies(controls)

"""
for base_directory in controls:
    psychophysical_interaction_workflow(base_directory, context_1_onset_files, context_2_onset_files, start_window, stop_window)

for base_directory in mutants:
    psychophysical_interaction_workflow(base_directory, context_1_onset_files, context_2_onset_files, start_window, stop_window)
"""


view_modulation_matricies(controls)
view_modulation_matricies(mutants)
