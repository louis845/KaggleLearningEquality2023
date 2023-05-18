import numpy as np
import sklearn
import sklearn.manifold
import matplotlib.pyplot as plt

import os
import gc
import time

def load_data(data_folder, data_to_convert):
    assert os.path.isdir(data_folder), "The data folder does not exist."
    assert os.path.isdir(os.path.join(data_folder, data_to_convert)), "The data to convert does not exist."

    # Load data, we only convert the topics, not the contents.
    if os.path.isfile(os.path.join(data_folder, data_to_convert, "topics.npy")):
        topics_data = np.load(os.path.join(data_folder, data_to_convert, "topics.npy"))
    elif os.path.isfile(os.path.join(data_folder, data_to_convert, "topics_description.npy")):
        assert os.path.isfile(os.path.join(data_folder, data_to_convert, "topics_title.npy")), "Incomplete data."
        topics_data = np.concatenate((np.load(os.path.join(data_folder, data_to_convert, "topics_title.npy")),
                                      np.load(os.path.join(data_folder, data_to_convert, "topics_description.npy"))),
                                     axis=1)
    print(topics_data.dtype)
    print("Loaded topics data with shape: ", topics_data.shape)
    return topics_data

def isomap_embedding_plot(data_folder="generated_data", data_to_convert="mininet_L12_english384", figsize=(10, 10)):
    topics_data = load_data(data_folder, data_to_convert)
    # Loop through n_components [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ..., 150]
    components = np.arange(20, 320, 20)
    avg_errors = np.zeros(shape=len(components), dtype=np.float32)
    rng = np.random.default_rng()

    for n_components in components:
        errors = np.zeros(shape=5, dtype=np.float32)
        ctime = time.time()
        for k in range(5):
            isomap = sklearn.manifold.Isomap(n_components=n_components, n_neighbors=20, n_jobs=-1)
            topics_data_sampled = topics_data[rng.choice(topics_data.shape[0], 10000, replace=False), :]
            isomap.fit(topics_data_sampled)
            errors[k] = isomap.reconstruction_error()
            del isomap, topics_data_sampled
            gc.collect()
            if n_components == 20 and k == 0:
                print("Finished first isomap in ", time.time() - ctime, " seconds.")
        avg_errors[int(n_components / 20) - 1] = np.mean(errors)
        print("Finished n_components = ", n_components, " in ", time.time() - ctime, " seconds.")

    # Plot the average reconstruction error vs the number of components
    plt.figure(figsize=figsize)
    plt.plot(components, avg_errors)
    plt.xlabel("n_components (dimension)")
    plt.ylabel("Average reconstruction error")
    plt.title("Isomap reconstruction error vs dimension")
    plt.show()

def ltsa_embedding_plot(data_folder="generated_data", data_to_convert="mininet_L12_english384", figsize=(10, 10)):
    topics_data = load_data(data_folder, data_to_convert)
    # Loop through n_components [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ..., 150]
    components = np.arange(20, 320, 20)
    avg_errors = np.zeros(shape=len(components), dtype=np.float32)
    rng = np.random.default_rng()

    for n_components in components:
        errors = np.zeros(shape=5, dtype=np.float32)
        ctime = time.time()
        for k in range(3):
            ltsa = sklearn.manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_components*2, method="ltsa", n_jobs=4)
            topics_data_sampled = topics_data[rng.choice(topics_data.shape[0], 5000, replace=False), :]
            ltsa.fit(topics_data_sampled)
            errors[k] = ltsa.reconstruction_error_
            del ltsa, topics_data_sampled
            gc.collect()
            if n_components == 20 and k == 0:
                print("Finished first ltsa in ", time.time() - ctime, " seconds.")
        avg_errors[int(n_components / 20) - 1] = np.mean(errors)
        print("Finished n_components = ", n_components, " in ", time.time() - ctime, " seconds.")

    # Plot the average reconstruction error vs the number of components
    plt.figure(figsize=figsize)
    plt.plot(components, avg_errors)
    plt.xlabel("n_components (dimension)")
    plt.ylabel("Average reconstruction error")
    plt.title("LTSA reconstruction error vs dimension")
    plt.show()