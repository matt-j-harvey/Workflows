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




def psychophysical_interaction_workflow(base_directory, onset_files, context_1_tensor_names, context_2_tensor_names, start_window, stop_window, selected_region_names, region_group_names):

    # Check Output Folders Exist
    functional_connectivity_directory = os.path.join(base_directory, "Functional Connectivity Analysis")
    ppi_save_directory = os.path.join(functional_connectivity_directory, "Psychophysical Interactions")
    check_directory(functional_connectivity_directory)
    check_directory(ppi_save_directory)

    # Create Tensors
    number_of_conditions = len(onset_files)
    tensor_names = context_1_tensor_names + context_2_tensor_names
    for condition_index in range(number_of_conditions):
        Create_Activity_Tensor.create_activity_tensor(base_directory, onset_files[condition_index], start_window, stop_window, tensor_names[condition_index])


    # Load Pixel Assignments
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")

    # Create Comparison Images
    indicies, height, width = Widefield_General_Functions.load_mask(base_directory)
    number_of_pixels = len(indicies)


    number_of_conditions = [len(context_1_tensor_names), len(context_2_tensor_names)]

    # Iterate Through Each Region Group
    number_of_regions = len(selected_region_names)
    for region_index in range(number_of_regions):

        region_group = selected_region_names[region_index]

        # Get Atlas Labels
        selected_region_labels = load_atlas_labels(region_group)
        print("Selected reigon labels", selected_region_labels)

        # Load Activity Tensors
        context_1_activity_tensor_list = []
        for condition_index in range(number_of_conditions[0]):
            activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", context_1_tensor_names[condition_index] + "_Activity_Tensor.npy"))
            activity_tensor = np.nan_to_num(activity_tensor)
            context_1_activity_tensor_list.append(activity_tensor)

        context_2_activity_tensor_list = []
        for condition_index in range(number_of_conditions[1]):
            activity_tensor = np.load(os.path.join(base_directory, "Activity_Tensors", context_2_tensor_names[condition_index] + "_Activity_Tensor.npy"))
            activity_tensor = np.nan_to_num(activity_tensor)
            context_2_activity_tensor_list.append(activity_tensor)



        # Get Region Mean Tensors
        context_1_region_tensor_list = []
        for condition_index in range(number_of_conditions[0]):
            region_tensor = Get_Region_Mean_Tensor.get_region_mean_tensor(context_1_activity_tensor_list[condition_index], pixel_assignments, selected_region_labels)
            context_1_region_tensor_list.append(region_tensor)

        context_2_region_tensor_list = []
        for condition_index in range(number_of_conditions[1]):
            region_tensor = Get_Region_Mean_Tensor.get_region_mean_tensor(context_2_activity_tensor_list[condition_index], pixel_assignments, selected_region_labels)
            context_2_region_tensor_list.append(region_tensor)

        # Perform Noise Correlation Analysis
        ppi_coefs = Perform_Psychophysical_Interaction_Analysis.create_ppi_model(context_1_activity_tensor_list, context_2_activity_tensor_list, context_1_region_tensor_list, context_2_region_tensor_list)

        print("PPI Coes", ppi_coefs)
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
onset_files = [["visual_context_stable_vis_1_onsets.npy"], ["visual_context_stable_vis_2_onsets.npy"], ["odour_context_stable_vis_1_onsets.npy"],  ["odour_context_stable_vis_2_onsets.npy"]]
context_1_tensor_names = ["Vis_1_Stable_Visual", "Vis_2_Stable_Visual"]
context_2_tensor_names = ["Vis_1_Stable_Odour", "Vis_2_Stable_Odour"]

selected_regions = [['Primary_Visual_Left', 'Primary_Visual_Right'],
                    ['Retosplenlial_Dorsal'],
                    ['Secondary_Motor_Left','Secondary_Motor_Right'],
                    ['Primary_Somatosensory_Barrel_Left', 'Primary_Somatosensory_Barrel_Right']]


region_group_names = ["Prinary_Visual", "Retrosplenial", "Secondary_Motor", "Somatosensory"]


#all_mice = ["/media/matthew/Seagate Expansion Drive1/Widefield_Imaging/Transition_Analysis/NXAK4.1B/2021_04_10_Transition_Imaging"]
for base_directory in all_mice:
    psychophysical_interaction_workflow(base_directory, onset_files, context_1_tensor_names, context_2_tensor_names, start_window, stop_window, selected_regions, region_group_names)

get_significant_modulation(controls)

