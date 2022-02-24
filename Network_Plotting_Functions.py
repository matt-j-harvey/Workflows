import networkx as nxx
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import os
from scipy import stats
import networkx as nx
from matplotlib.pyplot import cm

def get_region_centroids():

    # Load Outline Array
    #cluster_outlines = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy")

    # Load Regions

    pixel_assignment_image = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")

    image_height = np.shape(pixel_assignment_image)[0]

    unqiue_regions = np.unique(pixel_assignment_image)
    print(unqiue_regions)


    centroids = []
    for region in unqiue_regions:
        region_mask = np.where(pixel_assignment_image == region, 1, 0)
        region_mask_indicies = np.nonzero(region_mask)


        region_y_mean = int(np.mean(region_mask_indicies[0]))
        region_x_mean = int(np.mean(region_mask_indicies[1]))

        centroids.append([region_x_mean, region_y_mean])
    centroids = np.array(centroids)
    np.save("/home/matthew/Documents/Allen_Atlas_Templates/New_Centroids.npy", centroids)

    print("Centroids ", centroids)

    plt.imshow(pixel_assignment_image)
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.show()

    mask = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/mask.npy")


    outline = np.where(pixel_assignment_image == 0, 1, 0)
    outline = np.multiply(outline, mask)
    outline = np.where(outline > 0.00000001, 1, 0)
    outline = feature.canny(pixel_assignment_image, sigma=3)

    plt.imshow(outline)
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.show()

    np.save("/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy", outline)


def draw_brain_network(adjacency_matrix):

    # Load Allen Region Outlines
    cluster_outlines = np.load("/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy")
    image_height = np.shape(cluster_outlines)[0]


    # Load Allen Region Centroids
    cluster_centroids = np.load("/home/matthew/Documents/Allen_Atlas_Templates/New_Centroids.npy")
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])
    inverted_centroids = np.array(inverted_centroids)

    # Draw Network
    plt.imshow(cluster_outlines)
    plt.scatter(inverted_centroids[:, 0], inverted_centroids[:, 1])
    plt.show()

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph())



    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    weights = np.divide(weights, np.max(np.abs(weights)))

    # Get Edge Colours
    positive_colourmap = cm.get_cmap('Reds')
    negative_colourmap = cm.get_cmap('Blues')

    colours = []
    for weight in weights:

        if weight >= 0:
            colour = positive_colourmap(weight)
        else:
            colour = negative_colourmap(np.abs(weight))

        colours.append(colour)




    # Draw Graph
    # Invert Cluster Centroids

    #plt.title(session_name)
    plt.imshow(cluster_outlines, cmap='binary')
    nx.draw(graph, pos=cluster_centroids, node_size=1, edge_color=colours, width=weights, arrowsize=10) #,create_using=nx.MultiDiGraph()
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()




def draw_brain_network_undirected(adjacency_matrix):

    # Load Allen Region Outlines
    cluster_outlines = np.load("/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy")
    image_height = np.shape(cluster_outlines)[0]


    # Load Allen Region Centroids
    cluster_centroids = np.load("/home/matthew/Documents/Allen_Atlas_Templates/New_Centroids.npy")
    inverted_centroids = []
    for centroid in cluster_centroids:
        y_value = centroid[1]
        x_value = centroid[0]
        inverted_y = image_height - y_value
        inverted_centroids.append([x_value, inverted_y])
    inverted_centroids = np.array(inverted_centroids)

    # Draw Network
    plt.imshow(cluster_outlines)
    plt.scatter(inverted_centroids[:, 0], inverted_centroids[:, 1])
    plt.show()

    # Create NetworkX Graph
    graph = nx.from_numpy_matrix(adjacency_matrix)



    # Get Edge Weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]

    if np.sum(np.abs(weights)) > 0.1:
        weights = np.divide(weights, np.max(np.abs(weights)))

    # Get Edge Colours
    positive_colourmap = cm.get_cmap('Reds')
    negative_colourmap = cm.get_cmap('Blues')

    colours = []
    for weight in weights:

        if weight >= 0:
            colour = positive_colourmap(weight)
        else:
            colour = negative_colourmap(np.abs(weight))

        colours.append(colour)




    # Draw Graph
    # Invert Cluster Centroids

    #plt.title(session_name)
    plt.imshow(cluster_outlines, cmap='binary')
    nx.draw(graph, pos=cluster_centroids, node_size=1, edge_color=colours, width=weights) #,create_using=nx.MultiDiGraph()
    plt.show()
    #plt.savefig(base_directory + "/" + session_name + "_Signficant_Correlation_Changes.png")
    #plt.close()



def plot_beta_weight_correlations(session_list):

    context_1_matrix_list = []
    context_2_matrix_list = []
    modulation_matrix_list = []

    for base_directory in session_list:
        context_1_matrix = np.load(os.path.join(base_directory, "Functional Connectivity Analysis", "Beta Series Correlations", "Context_1_Correlation_Map.npy"))
        context_2_matrix = np.load(os.path.join(base_directory, "Functional Connectivity Analysis", "Beta Series Correlations", "Context_2_Correlation_Map.npy"))
        difference_matrix = np.subtract(context_1_matrix, context_2_matrix)

        context_1_matrix_list.append(context_1_matrix)
        context_2_matrix_list.append(context_2_matrix)
        modulation_matrix_list.append(difference_matrix)

        context_1_matrix_magnitude = np.max(np.abs(context_1_matrix))
        context_2_matrix_magnitude = np.max(np.abs(context_2_matrix))
        difference_matrix_magnitude = np.max(np.abs(difference_matrix))

        """
        rows = 1
        columns = 3
        figure_1 = plt.figure()
        context_1_axis = figure_1.add_subplot(rows, columns, 1)
        context_2_axis = figure_1.add_subplot(rows, columns, 2)
        differnece_axis = figure_1.add_subplot(rows, columns, 3)

        print("context 2 mstrix shape", np.shape(context_1_matrix))
        context_1_axis.imshow(context_1_matrix, cmap='bwr', vmin=-1 * context_1_matrix_magnitude, vmax=context_1_matrix_magnitude)
        context_2_axis.imshow(context_2_matrix, cmap='bwr', vmin=-1 * context_2_matrix_magnitude, vmax=context_2_matrix_magnitude)
        differnece_axis.imshow(difference_matrix, cmap='bwr', vmin=-1 * difference_matrix_magnitude, vmax=difference_matrix_magnitude)
        plt.show()
    """
    modulation_matrix_list = np.array(modulation_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matrix_list, axis=0)

    plt.title("Unthresholded Beta Series Weights")
    plt.imshow(mean_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()

    # mean_modulation_matrix = np.add(mean_modulation_matrix, np.transpose(mean_modulation_matrix))
    t_stats, p_values = stats.ttest_rel(context_1_matrix_list, context_2_matrix_list)

    p_threshold = 0.05
    thresholded_modulation_matrix = np.where(p_values < p_threshold, mean_modulation_matrix, 0)

    plt.title("thresholded Beta Series Weights")
    plt.imshow(thresholded_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()


    draw_brain_network(mean_modulation_matrix)
    draw_brain_network(thresholded_modulation_matrix)

#get_region_centroids()

def plot_ppi_matricies(session_list):

    context_1_matrix_list = []
    context_2_matrix_list = []
    modulation_matrix_list = []

    for base_directory in session_list:
        ppi_coef_meta_list = np.load(os.path.join(base_directory, "Functional Connectivity Analysis/Psychophysical Interactions/PPI_Coefs_All_Regions.npy"))
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

        """
        context_1_matrix_magnitude = np.max(np.abs(context_1_matrix))
        context_2_matrix_magnitude = np.max(np.abs(context_2_matrix))
        difference_matrix_magnitude = np.max(np.abs(difference_matrix))

        # difference_matrix, [context_1_matrix, context_2_matrix] = jointly_sort_matricies(difference_matrix, [context_1_matrix, context_2_matrix])

        rows = 1
        columns = 3
        figure_1 = plt.figure()
        context_1_axis = figure_1.add_subplot(rows, columns, 1)
        context_2_axis = figure_1.add_subplot(rows, columns, 2)
        differnece_axis = figure_1.add_subplot(rows, columns, 3)

        print("context 2 mstrix shape", np.shape(context_1_matrix))
        context_1_axis.imshow(context_1_matrix, cmap='bwr', vmin=-1 * context_1_matrix_magnitude, vmax=context_1_matrix_magnitude)
        context_2_axis.imshow(context_2_matrix, cmap='bwr', vmin=-1 * context_2_matrix_magnitude, vmax=context_2_matrix_magnitude)
        differnece_axis.imshow(difference_matrix, cmap='bwr', vmin=-1 * difference_matrix_magnitude, vmax=difference_matrix_magnitude)

        figure_1.suptitle(base_directory.split('/')[-2:])
        plt.show()
    """
    modulation_matrix_list = np.array(modulation_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matrix_list, axis=0)

    mean_modulation_matrix = exclude_regions(mean_modulation_matrix)
    mean_modulation_matrix = np.transpose(mean_modulation_matrix)

    plt.title("Unthresholded PPI Connectivity")
    plt.imshow(mean_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()

    # anyzmean_modulation_matrix = np.add(mean_modulation_matrix, np.transpose(mean_modulation_matrix))
    t_stats, p_values = stats.ttest_rel(context_1_matrix_list, context_2_matrix_list)

    p_threshold = 0.05
    thresholded_modulation_matrix = np.where(p_values < p_threshold, mean_modulation_matrix, 0)

    plt.title("Thresholded PPI Connectivity")
    plt.imshow(thresholded_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()


    # Split Into Positives
    positive_modulation_matrix = np.where(thresholded_modulation_matrix > 0, thresholded_modulation_matrix, 0)
    negative_modulation_matrix = np.where(thresholded_modulation_matrix < 0, thresholded_modulation_matrix, 0)


    draw_brain_network(positive_modulation_matrix)
    draw_brain_network(negative_modulation_matrix)


    draw_brain_network(mean_modulation_matrix)
    draw_brain_network(thresholded_modulation_matrix)


def plot_beta_weight_matricies(session_list):

    context_1_matrix_list = []
    context_2_matrix_list = []
    modulation_matrix_list = []

    for base_directory in session_list:
        context_1_matrix = np.load(os.path.join(base_directory, "Functional Connectivity Analysis", "Beta Series Correlations", "Context_1_Correlation_Map.npy"))
        context_2_matrix = np.load(os.path.join(base_directory, "Functional Connectivity Analysis", "Beta Series Correlations", "Context_2_Correlation_Map.npy"))
        difference_matrix = np.subtract(context_1_matrix, context_2_matrix)
        #difference_matrix = np.subtract(context_2_matrix, context_1_matrix)


        context_1_matrix_list.append(context_1_matrix)
        context_2_matrix_list.append(context_2_matrix)
        modulation_matrix_list.append(difference_matrix)

        """
        context_1_matrix_magnitude = np.max(np.abs(context_1_matrix))
        context_2_matrix_magnitude = np.max(np.abs(context_2_matrix))
        difference_matrix_magnitude = np.max(np.abs(difference_matrix))

        # difference_matrix, [context_1_matrix, context_2_matrix] = jointly_sort_matricies(difference_matrix, [context_1_matrix, context_2_matrix])

        rows = 1
        columns = 3
        figure_1 = plt.figure()
        context_1_axis = figure_1.add_subplot(rows, columns, 1)
        context_2_axis = figure_1.add_subplot(rows, columns, 2)
        differnece_axis = figure_1.add_subplot(rows, columns, 3)

        print("context 2 mstrix shape", np.shape(context_1_matrix))

        figure_1.suptitle(base_directory.split('/')[-2:])
        context_1_axis.imshow(context_1_matrix, cmap='bwr', vmin=-1 * context_1_matrix_magnitude, vmax=context_1_matrix_magnitude)
        context_2_axis.imshow(context_2_matrix, cmap='bwr', vmin=-1 * context_2_matrix_magnitude, vmax=context_2_matrix_magnitude)
        differnece_axis.imshow(difference_matrix, cmap='bwr', vmin=-1 * difference_matrix_magnitude, vmax=difference_matrix_magnitude)
        plt.show()
        """

    modulation_matrix_list = np.array(modulation_matrix_list)
    mean_modulation_matrix = np.mean(modulation_matrix_list, axis=0)


    mean_modulation_matrix = exclude_regions(mean_modulation_matrix)


    plt.title("Unthresholded Beta Series Weights")
    plt.imshow(mean_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()

    # mean_modulation_matrix = np.add(mean_modulation_matrix, np.transpose(mean_modulation_matrix))
    t_stats, p_values = stats.ttest_rel(context_1_matrix_list, context_2_matrix_list)

    p_threshold = 0.05
    thresholded_modulation_matrix = np.where(p_values < p_threshold, mean_modulation_matrix, 0)

    plt.title("thresholded Beta Series Weights")
    plt.imshow(thresholded_modulation_matrix, cmap='bwr', vmin=-1 * np.max(np.abs(mean_modulation_matrix)), vmax=np.max(np.abs(mean_modulation_matrix)))
    plt.show()

    # Split Into Positives
    positive_modulation_matrix = np.where(thresholded_modulation_matrix > 0, thresholded_modulation_matrix, 0)
    negative_modulation_matrix = np.where(thresholded_modulation_matrix < 0, thresholded_modulation_matrix, 0)


    draw_brain_network_undirected(positive_modulation_matrix)
    draw_brain_network_undirected(negative_modulation_matrix)


    draw_brain_network_undirected(mean_modulation_matrix)
    draw_brain_network_undirected(thresholded_modulation_matrix)



def exclude_regions(adjacency_matrix):

    # Get Exclusion List
    exclusion_list = [0, 1, 10, 5, 3, 4, 17, 18]

    # get Pixel Assignemnt List
    pixel_assignments = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets.npy")
    cluster_list = list(np.unique(pixel_assignments))

    for region in exclusion_list:
        region_index = cluster_list.index(region)
        adjacency_matrix[region_index] = 0
        adjacency_matrix[:, region_index] = 0

    return adjacency_matrix



def view_pixel_assignments():
    pixel_assignments_image = np.load("/media/matthew/Seagate Expansion Drive/Widefield_Imaging/Transition_Analysis/NXAK16.1B/2021_07_08_Transition_Imaging/Pixel_Assignmnets_Image.npy")
    plt.imshow(pixel_assignments_image)
    plt.show()


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
            "/media/matthew/Seagate Expansion Drive1/Transition_2/NXAK20.1B/2021_11_26_Transition_Imaging"
            ]


view_pixel_assignments()
#plot_beta_weight_correlations(controls)
#plot_ppi_matricies(controls)

#plot_beta_weight_matricies(controls)
#plot_beta_weight_matricies(mutants)

plot_ppi_matricies(controls)
#plot_ppi_matricies(mutants)