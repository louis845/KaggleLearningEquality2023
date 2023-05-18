import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

import CSIC_classifier
import CSIC_data_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(reduction_type, n_neighbors, neighbor_type, noise):
    """

    :param reduction_type: isomap, pca, None
    :param n_neighbors: 4, 8, 16 for isomap or pca. Ignored for None (use original text embedding by transformer)
    :param neighbor_type: similar or dissimilar. Ignored for None.
    :param noise: Noise
    :return:
    """

    if reduction_type is None:
        model_dir = os.path.join("saved_models", "default_noise_{}".format(noise))
        topic_embedding_file = None
    else:
        model_dir = os.path.join("saved_models", "{}_{}neighbors_{}_noise_{}".format(reduction_type, n_neighbors, neighbor_type, noise))
        topic_embedding_file = os.path.join("generated_data", "mininet_L12_english384", "topics_neighbors_{}_{}_{}.npy".format(reduction_type, n_neighbors, neighbor_type))

    contents_embeddings = np.concatenate([np.load("generated_data/mininet_L12_english384/contents_title.npy"),
                                          np.load(
                                              "generated_data/mininet_L12_english384/contents_description.npy")],
                                         axis=1)
    if topic_embedding_file is None:
        topics_embeddings = np.concatenate([np.load("generated_data/mininet_L12_english384/topics_title.npy"),
                                            np.load(
                                                "generated_data/mininet_L12_english384/topics_description.npy")],
                                           axis=1)
    else:
        topics_embeddings = np.load(topic_embedding_file)

    contents_embeddings = torch.tensor(contents_embeddings, dtype=torch.float32, device=device)
    topics_embeddings = torch.tensor(topics_embeddings, dtype=torch.float32, device=device)

    test_batch_size, num_iters = 32768, 5

    # load the model
    model = CSIC_classifier.Classifier(contents_variance=0.0, topics_variance=0.0,
                                       input_gaussian_noise=0.0, hidden_dropout_rate=0.0).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    with torch.no_grad():
        total, total_correct = 0, 0
        for k in range(num_iters):
            topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(test_batch_size // 2, test_batch_size // 2)
            while np.sum(cor) == 0:
                topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(test_batch_size // 2, test_batch_size // 2)
            contents_embeddings_batch = contents_embeddings[contents_num_id, :]
            topics_embeddings_batch = topics_embeddings[topics_num_id, :]
            labels_batch = torch.tensor(cor, dtype=torch.float32, device=device)

            outputs = torch.squeeze(
                model(torch.cat([contents_embeddings_batch, topics_embeddings_batch], dim=1), eval=True),
                dim=1)

            actual_labels = cor.astype(dtype=np.int32)
            predicted_labels = outputs.cpu().numpy() > 0.5

            # Compute accuracy. Actual labels and predicted labels are integers either 0 or 1.
            total += len(actual_labels)
            total_correct += np.sum(actual_labels == predicted_labels)
        accuracy_best_model = total_correct / total

    del model
    model = CSIC_classifier.Classifier(contents_variance=0.0, topics_variance=0.0,
                                       input_gaussian_noise=0.0, hidden_dropout_rate=0.0).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "final_epoch.pt"), map_location=device))
    with torch.no_grad():
        total, total_correct = 0, 0
        for k in range(num_iters):
            topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(test_batch_size // 2,
                                                                                    test_batch_size // 2)
            while np.sum(cor) == 0:
                topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(test_batch_size // 2,
                                                                                        test_batch_size // 2)
            contents_embeddings_batch = contents_embeddings[contents_num_id, :]
            topics_embeddings_batch = topics_embeddings[topics_num_id, :]
            labels_batch = torch.tensor(cor, dtype=torch.float32, device=device)

            outputs = torch.squeeze(
                model(torch.cat([contents_embeddings_batch, topics_embeddings_batch], dim=1), eval=True),
                dim=1)

            actual_labels = cor.astype(dtype=np.int32)
            predicted_labels = outputs.cpu().numpy() > 0.5

            # Compute accuracy. Actual labels and predicted labels are integers either 0 or 1.
            total += len(actual_labels)
            total_correct += np.sum(actual_labels == predicted_labels)
        accuracy_250_epochs = total_correct / total

    del model, contents_embeddings, topics_embeddings
    return accuracy_best_model, accuracy_250_epochs

summary_text = []
summary_best_performance = []

models_dir = "saved_models"

best_model_accuracy_no_reduction = []
final_epoch_accuracy_no_reduction = []
for k in range(6):
    print("Running noise {}".format(10 * k))
    ctime = time.time()
    accuracy_best_model, accuracy_250_epochs = get_accuracy(None, None, None, 10 * k)
    best_model_accuracy_no_reduction.append(accuracy_best_model)
    final_epoch_accuracy_no_reduction.append(accuracy_250_epochs)
    print("Time taken: {}".format(time.time() - ctime))

summary_text.append("No neighbors best model")
summary_best_performance.append(np.max(best_model_accuracy_no_reduction))
summary_text.append("No neighbors final epoch")
summary_best_performance.append(np.max(final_epoch_accuracy_no_reduction))


best_model_accuracy = []
final_epoch_accuracy = []
reduction_type = "isomap"
for n_neighbors in [4, 8, 16]:
    for neighbor_type in ["similar", "dissimilar"]:
        best_model_accuracy.clear()
        final_epoch_accuracy.clear()
        for k in range(6):
            print("Running {} {} {} noise {}".format(reduction_type, n_neighbors, neighbor_type, 2 * k))
            ctime = time.time()
            accuracy_best_model, accuracy_250_epochs = get_accuracy(reduction_type, n_neighbors, neighbor_type, 2 * k)
            best_model_accuracy.append(accuracy_best_model)
            final_epoch_accuracy.append(accuracy_250_epochs)
            print("Time taken: {}".format(time.time() - ctime))

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        l1 = ax.plot(np.arange(0, 60, 10), best_model_accuracy_no_reduction, label="No reduction best model", color="blue")
        l2 = ax.plot(np.arange(0, 60, 10), final_epoch_accuracy_no_reduction, label="No reduction final epoch", color="red")
        ax_twin = ax.twiny()
        l3 = ax_twin.plot(np.arange(6) * 2, best_model_accuracy, label="{} {} {} best model".format(reduction_type, n_neighbors, neighbor_type), color="green")
        l4 = ax_twin.plot(np.arange(6) * 2, final_epoch_accuracy, label="{} {} {} final epoch".format(reduction_type, n_neighbors, neighbor_type), color="orange")


        ax.set_xlabel("Noise level")
        ax.set_ylabel("Accuracy")
        ax_twin.set_xlabel("Noise level")

        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc=0)

        fig.savefig(os.path.join(models_dir, "{}_{}_{}.png".format(reduction_type, n_neighbors, neighbor_type)))


        summary_text.append("{} {} {} best model".format(reduction_type, n_neighbors, neighbor_type))
        summary_best_performance.append(np.max(best_model_accuracy))
        summary_text.append("{} {} {} final epoch".format(reduction_type, n_neighbors, neighbor_type))
        summary_best_performance.append(np.max(final_epoch_accuracy))

reduction_type = "pca"
for n_neighbors in [4, 8, 16]:
    for neighbor_type in ["similar", "dissimilar"]:
        best_model_accuracy.clear()
        final_epoch_accuracy.clear()
        for k in range(6):
            print("Running {} {} {} noise {}".format(reduction_type, n_neighbors, neighbor_type, 10 * k))
            ctime = time.time()
            accuracy_best_model, accuracy_250_epochs = get_accuracy(reduction_type, n_neighbors, neighbor_type, 10 * k)
            best_model_accuracy.append(accuracy_best_model)
            final_epoch_accuracy.append(accuracy_250_epochs)
            print("Time taken: {}".format(time.time() - ctime))

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        ax.plot(np.arange(0, 60, 10), best_model_accuracy_no_reduction, label="No reduction best model", color="blue")
        ax.plot(np.arange(0, 60, 10), final_epoch_accuracy_no_reduction, label="No reduction final epoch", color="red")
        ax.plot(np.arange(0, 60, 10), best_model_accuracy,
                     label="{} {} {} best model".format(reduction_type, n_neighbors, neighbor_type), color="green")
        ax.plot(np.arange(0, 60, 10), final_epoch_accuracy,
                     label="{} {} {} final epoch".format(reduction_type, n_neighbors, neighbor_type), color="orange")

        ax.set_xlabel("Noise level")
        ax.set_ylabel("Accuracy")

        ax.legend()

        fig.savefig(os.path.join(models_dir, "{}_{}_{}.png".format(reduction_type, n_neighbors, neighbor_type)))

        summary_text.append("{} {} {} best model".format(reduction_type, n_neighbors, neighbor_type))
        summary_best_performance.append(np.max(best_model_accuracy))
        summary_text.append("{} {} {} final epoch".format(reduction_type, n_neighbors, neighbor_type))
        summary_best_performance.append(np.max(final_epoch_accuracy))


# Use pandas to create a table of the results, and save it to "saved_models/summary.csv"
summary = pd.DataFrame({"Model": summary_text, "Best accuracy (over noise)": summary_best_performance})
summary.to_csv("saved_models/summary.csv", index=False)