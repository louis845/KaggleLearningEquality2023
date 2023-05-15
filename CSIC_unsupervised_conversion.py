import numpy as np
import sklearn
import sklearn.manifold
import sklearn.decomposition

import argparse
import os
import time

# Parse arguments
parser = argparse.ArgumentParser(description='CSIC unsupervised conversion')
# Add argument data folder, default is generated_data
parser.add_argument('--data_folder', type=str, default="generated_data",
                    help='folder containing the data')
# Add argument data to convert, default is mininet_L12_english384
parser.add_argument('--data_to_convert', type=str, default="mininet_L12_english384",
                    help='data to convert')
# Add argument dimension reduction method, default is PCA
parser.add_argument('--dimension_reduction_method', type=str, default="PCA",
                    help='dimension reduction method (PCA/isomap/LLE)')
# Add argument target dimensions, default is 128
parser.add_argument('--target_dimensions', type=int, default=128,
                    help='target dimensions')

args = parser.parse_args()
data_folder = args.data_folder
data_to_convert = args.data_to_convert
dimension_reduction_method = args.dimension_reduction_method
target_dimensions = args.target_dimensions

if not os.path.isdir(data_folder):
    print("The data folder does not exist.")
    exit()
if not os.path.isdir(os.path.join(data_folder, data_to_convert)):
    print("The data to convert does not exist.")
    exit()

# Load data, we only convert the topics, not the contents.
if os.path.isfile(os.path.join(data_folder, data_to_convert, "topics.npy")):
    topics_data = np.load(os.path.join(data_folder, data_to_convert, "topics.npy"))
elif os.path.isfile(os.path.join(data_folder, data_to_convert, "topics_description.npy")):
    if not os.path.isfile(os.path.join(data_folder, data_to_convert, "topics_title.npy")):
        print("Incomplete data.")
        exit()
    topics_data = np.concatenate((np.load(os.path.join(data_folder, data_to_convert, "topics_title.npy")),
                                    np.load(os.path.join(data_folder, data_to_convert, "topics_description.npy"))), axis=1)

print("Loaded topics data with shape: ", topics_data.shape)
print("Reducing dimension {} ----> {} with method {}".format(topics_data.shape[1], target_dimensions, dimension_reduction_method))

ctime = time.time()
# Reduce dimension
if dimension_reduction_method == "PCA":
    pca = sklearn.decomposition.PCA(n_components=target_dimensions)
    topics_data_reduced = pca.fit_transform(topics_data)
elif dimension_reduction_method == "isomap":
    isomap = sklearn.manifold.Isomap(n_components=target_dimensions, n_neighbors=40)
    topics_data_reduced = isomap.fit_transform(topics_data)
elif dimension_reduction_method == "LLE":
    lle = sklearn.manifold.LocallyLinearEmbedding(n_components=target_dimensions, n_neighbors=40)
    topics_data_reduced = lle.fit_transform(topics_data)

print("Reduced dimension in {} seconds.".format(time.time() - ctime))

# Save data
np.save(os.path.join(data_folder, data_to_convert, "topics_reduced_{}_{}.npy".format(dimension_reduction_method, target_dimensions)), topics_data_reduced)