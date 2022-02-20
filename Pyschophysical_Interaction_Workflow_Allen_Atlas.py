import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats

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




def psychophysical_interaction_workflow(base_directory, context_1_onsets, context_2_onsets, start_window, stop_window, selected_region_names, region_group_names):

    # Load Pixel Assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))
    number_of_clusters = len(cluster_list)
    print("number of clusters", number_of_clusters)
    print("cluster list", cluster_list)

    # Show Pixel Assignmnent Image
    #pixel_assignment_image = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    #plt.imshow(pixel_assignment_image)
    #plt.show()

    # Check Output Folders Exist
    functional_connectivity_directory = os.path.join(base_directory, "Functional Connectivity Analysis")
    ppi_save_directory = os.path.join(functional_connectivity_directory, "Psychophysical Interactions")
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


    # Iterate Through Each Region Group
    number_of_regions = len(selected_region_names)
    for region_index in range(number_of_regions):

        region_group = selected_region_names[region_index]

        # Get Atlas Labels
        selected_region_labels = load_atlas_labels(region_group)

        # Convert Region Labels To Indexes
        region_index_list = []
        for region_label in selected_region_labels:
            print("label", region_label)
            allen_region_index = cluster_list.index(region_label)
            print("Index", allen_region_index)
            region_index_list.append(allen_region_index)
        region_index_list.sort()

        print("Selected reigon labels", selected_region_labels)

        # Create Region Tensors
        context_1_region_tensor_list = []
        for condition_index in range(number_of_context_1_conditions):
            activity_tensor = context_1_activity_tensor_list[condition_index]
            region_tensor = activity_tensor[:, :, region_index_list]

            # If We are using Bilateral Regions
            if len(selected_region_labels) > 1:
                region_tensor = np.mean(region_tensor, axis=2)

            context_1_region_tensor_list.append(region_tensor)


        context_2_region_tensor_list = []
        for condition_index in range(number_of_context_2_conditions):
            activity_tensor = context_2_activity_tensor_list[condition_index]
            region_tensor = activity_tensor[:, :, region_index_list]

            # If We are using Bilateral Regions
            if len(selected_region_labels) > 1:
                region_tensor = np.mean(region_tensor, axis=2)

            context_2_region_tensor_list.append(region_tensor)


        # Check Shape
        for tensor in context_1_activity_tensor_list:
            print("Activity Tensor Shape", np.shape(tensor))

        for tensor in context_1_region_tensor_list:
            print("Region Tensor Shape", np.shape(tensor))


        # Perform Noise Correlation Analysis
        ppi_coefs = Perform_Psychophysical_Interaction_Analysis.create_ppi_model(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list)

        print("PPI Coes")
        print(np.shape(ppi_coefs))
        # Save PPI Coefficients
        ppi_coef_filename = region_group_names[region_index] + "_PPI_Coefs.npy"
        np.save(os.path.join(ppi_save_directory, ppi_coef_filename), ppi_coefs)

        # View This As An Image
        #image = Widefield_General_Functions.create_image_from_data(correlation_map, indicies, height, width)
        #print("Region tensor", np.shape(region_tensor))
    """

  
    for region_index in range(number_of_regions):

        # Load Correlation Maps
        correlation_maps = []
        for condition_index in range(number_of_conditions):
            correlation_map_filename = tensor_names[condition_index] + "_" + region_group_names[region_index] + "_Noise_Correlation_Map.npy"
            correlation_map = np.load(os.path.join(noise_correlation_save_directory, correlation_map_filename))
            correlation_maps.append(correlation_map)

        # Calculate Differences
        modulation_maps = np.zeros((number_of_conditions, number_of_conditions, number_of_pixels))
        for condition_1_index in range(number_of_conditions):
            for condition_2_index in range(number_of_conditions):
                modulation_map = np.subtract(correlation_maps[condition_1_index], correlation_maps[condition_2_index])
                modulation_maps[condition_1_index, condition_2_index] = modulation_map

        # Plot Differences
        figure_1 = plt.figure(figsize=(20,20))
        rows = number_of_conditions
        columns = number_of_conditions

        axis_count = 1
        for condition_1_index in range(number_of_conditions):
            for condition_2_index in range(number_of_conditions):

                modulation_map = modulation_maps[condition_1_index, condition_2_index]
                modulation_image = Widefield_General_Functions.create_image_from_data(modulation_map, indicies, height, width)
                image_magnitude = np.max(np.abs(modulation_map))

                axis = figure_1.add_subplot(rows, columns, axis_count)
                axis.imshow(modulation_image, cmap='bwr', vmax=image_magnitude, vmin=-1 * image_magnitude)
                axis.set_title(tensor_names[condition_1_index] + " v " + tensor_names[condition_2_index])
                axis.axis('off')

                axis_count += 1

        figure_1.suptitle(region_group_names[region_index])
        plt.savefig(os.path.join(noise_correlation_save_directory, "Noise Correlation Modulation.png"))
        plt.close()
      """
    # Create Comparison Matrix


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


mutants = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1A/2021_04_12_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK10.1A/2021_06_18_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK12.1F/2021_09_22_Transition_Imaging",
            "/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NRXN71.2A/2020_12_17_Switching_Imaging"]


all_mice = controls

start_window = -10
stop_window = 40
context_1_onset_files = [["visual_context_stable_vis_1_onsets.npy"], ["visual_context_stable_vis_2_onsets.npy"]]
context_2_onset_files = [["odour_context_stable_vis_1_onsets.npy"],  ["odour_context_stable_vis_2_onsets.npy"]]

selected_regions = [['Primary_Visual_Left', 'Primary_Visual_Right'],
                    ['Retosplenlial_Dorsal'],
                    ['Secondary_Motor_Left','Secondary_Motor_Right'],
                    ['Primary_Somatosensory_Barrel_Left', 'Primary_Somatosensory_Barrel_Right']]

region_group_names = ["Prinary_Visual", "Retrosplenial", "Secondary_Motor", "Somatosensory"]


#all_mice = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging"]
"""
for base_directory in all_mice:
    psychophysical_interaction_workflow(base_directory, context_1_onset_files, context_2_onset_files, start_window, stop_window, selected_regions, region_group_names)
"""
for base_directory in all_mice:
    view_ppi_modulation(base_directory, region_group_names[3])
    #view_ppi_coefs(base_directory, region_group_names[0], start_window, stop_window)
#get_significant_modulation(controls)

