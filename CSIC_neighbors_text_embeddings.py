import gc
import os

import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeighborMap():
    def __init__(self):
        self.graph = None
        self.max_neighbors = None
        self.min_neighbors = None
        self.distances = []

    def initiate(self):
        save_path = "generated_data/neighbor_map.npy"

        if os.path.exists(save_path):
            save_graph = np.load(save_path)
            self.graph = np.empty(len(save_graph), dtype="object")
            for i in range(save_graph.shape[0]):
                self.graph[i] = save_graph[i, save_graph[i] != -1]
            max_neighbors = max([len(x) for x in self.graph])
            del save_graph
            gc.collect()
        else:
            topics_df = pd.read_csv("generated_data/new_topics.csv", index_col = 0)
            str_id_to_int_id = pd.Series(np.arange(len(topics_df)), index = topics_df.index)

            self.graph = np.empty(len(topics_df), dtype="object")
            for i in range(len(topics_df)):
                self.graph[i] = [i]
            for i in range(len(topics_df)):
                if topics_df.iloc[i]["level"] > 0:
                    parent_id = str_id_to_int_id.loc[topics_df.loc[topics_df.index[i], "parent"]]
                    self.graph[i].append(parent_id)
                    self.graph[parent_id].append(i)
            for i in range(len(topics_df)):
                self.graph[i] = np.unique(np.array(self.graph[i], dtype=np.int32))

            # Convert to full numpy array and save
            max_neighbors = max([len(x) for x in self.graph])
            save_graph = np.full((len(self.graph), max_neighbors), dtype=np.int32, fill_value=-1)
            for i in range(len(self.graph)):
                save_graph[i, :len(self.graph[i])] = self.graph[i]
            np.save(save_path, save_graph)
            del save_graph
            gc.collect()
        self.max_neighbors = max_neighbors
        self.min_neighbors = min([len(x) for x in self.graph])

    def get_neighbors(self, node_id):
        return self.graph[node_id]

    def expand_distance_cache(self):
        self.distances.append(np.empty(len(self.graph), dtype="object"))
        if len(self.distances) == 1:
            for k in range(len(self.graph)):
                self.distances[0][k] = np.unique(np.concatenate([self.graph[neighbors_idx] for neighbors_idx in self.graph[k]]))
        else:
            for k in range(len(self.graph)):
                self.distances[-1][k] = np.unique(np.concatenate([self.distances[-2][neighbors_idx] for neighbors_idx in self.graph[k]]))

    def get_nodes_within_distance(self, node_id, distance):
        if distance == 1:
            return self.graph[node_id]
        return self.distances[distance - 2][node_id]

neighbors = None
def compute_neighbors_text_embeddings(topics_file="topics_reduced_isomap_48.npy", out_file="topics_neighbors_embeddings.npy", similar=True):
    """
    Compute the neighbors embeddings for the text embeddings.
    """
    global neighbors
    # Load topics embeddings
    topics_reduced_embeddings = np.load(os.path.join("generated_data/mininet_L12_english384", topics_file))
    topics_full_embeddings = torch.tensor(
        np.concatenate([np.load(os.path.join("generated_data/mininet_L12_english384", "topics_title.npy")),
                                            np.load(os.path.join("generated_data/mininet_L12_english384", "topics_description.npy"))])
    , dtype=torch.float32, device=device)
    print("Loaded topics embeddings with shape: ", topics_reduced_embeddings.shape)
    assert 768 % topics_reduced_embeddings.shape[1] == 0
    n_neighbors = 768 // topics_reduced_embeddings.shape[1]
    reduced_dim = topics_reduced_embeddings.shape[1]

    # Compute neighbors embeddings
    topics_neighbors_embeddings = np.zeros((topics_reduced_embeddings.shape[0], 768), dtype=np.float32)

    if neighbors is None:
        neighbors = NeighborMap()
        neighbors.initiate()
        for k in range(3):
            neighbors.expand_distance_cache()

    reduced_embeddings_mean = np.mean(topics_reduced_embeddings, axis=0)
    for i in range(topics_reduced_embeddings.shape[0]):
        if i % 10000 == 0:
            print("Computing neighbors for topic ", i)
        # Iteratively fill the array. The first reduced_dim values are the original reduced embedding (by PCA/isomap etc).
        # The remaining values are the reduced embeddings of neighbors wrt the graph. Precedence is given to closer neighbors.
        topics_neighbors_embeddings[i, :reduced_dim] = topics_reduced_embeddings[i, :]

        filled_values = 1
        distance = 1
        while filled_values < n_neighbors and distance < 5:
            neighbors_list_np = neighbors.get_nodes_within_distance(i, distance)
            if distance > 1:
                prev_neighbors = neighbors.get_nodes_within_distance(i, distance)
                # Remove previous neighbors from neighbors_list_np. neighbors_list_np and prev_neighbors are sorted. Use searchsorted
                neighbors_list_np = neighbors_list_np[np.searchsorted(prev_neighbors, neighbors_list_np, side="left") == np.searchsorted(prev_neighbors, neighbors_list_np, side="right")]
            elif distance == 1:
                neighbors_list_np = neighbors_list_np[neighbors_list_np != i]


            neighbors_list = torch.tensor(neighbors_list_np, dtype=torch.long, device=device)
            dist_squared = torch.sum((topics_full_embeddings[neighbors_list, :] - topics_full_embeddings[i, :]) ** 2, axis=1).cpu().numpy()
            sort = np.argsort(dist_squared)

            fill_values_end = min(filled_values + len(sort), n_neighbors)
            if similar:
                topics_neighbors_embeddings[i, reduced_dim * filled_values:reduced_dim * fill_values_end]\
                    = topics_reduced_embeddings[neighbors_list_np[sort[:fill_values_end - filled_values]], :].flatten()
            else:
                topics_neighbors_embeddings[i, reduced_dim * filled_values:reduced_dim * fill_values_end] \
                    = topics_reduced_embeddings[neighbors_list_np[sort[filled_values - fill_values_end:]], :].flatten()
            filled_values = fill_values_end
            distance += 1

        # Fill remaining values with the mean of the reduced embeddings for data imputation.
        if filled_values < n_neighbors:
            topics_neighbors_embeddings[i, reduced_dim * filled_values:reduced_dim * n_neighbors] = np.tile(reduced_embeddings_mean, n_neighbors - filled_values)

    np.save(os.path.join("generated_data/mininet_L12_english384", out_file), topics_neighbors_embeddings)


if __name__ == "__main__":
    print("Computing full embeddings for isomap reduction with 16 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_48.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 16, "similar"), similar=True)
    print("Computing full embeddings for isomap reduction with 8 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_96.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 8, "similar"), similar=True)
    print("Computing full embeddings for isomap reduction with 4 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_192.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 4, "similar"), similar=True)
    print("Computing full embeddings for isomap reduction with 16 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_48.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 16, "dissimilar"), similar=False)
    print("Computing full embeddings for isomap reduction with 8 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_96.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 8, "dissimilar"), similar=False)
    print("Computing full embeddings for isomap reduction with 4 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_isomap_192.npy", "topics_neighbors_{}_{}_{}.npy".format("isomap", 4, "dissimilar"), similar=False)

    print("Computing full embeddings for pca reduction with 16 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_48.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 16, "similar"), similar=True)
    print("Computing full embeddings for pca reduction with 8 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_96.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 8, "similar"), similar=True)
    print("Computing full embeddings for pca reduction with 4 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_192.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 4, "similar"), similar=True)
    print("Computing full embeddings for pca reduction with 16 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_48.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 16, "dissimilar"), similar=False)
    print("Computing full embeddings for pca reduction with 8 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_96.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 8, "dissimilar"), similar=False)
    print("Computing full embeddings for pca reduction with 4 neighbors")
    compute_neighbors_text_embeddings("topics_reduced_pca_192.npy", "topics_neighbors_{}_{}_{}.npy".format("pca", 4, "dissimilar"), similar=False)