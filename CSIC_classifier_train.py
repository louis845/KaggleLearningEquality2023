import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import CSIC_data_bert
import CSIC_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierTraining():
    def __init__(self, input_gaussian_noise, batch_size=32, test_batch_size=2048, lr=0.0001, dropout_rate=0.1, topic_embedding_file=None, save_path="saved_models/CSIC_classifier.pt"):
        """
        Train classifier on embeddings and labels. Save model to save_path.
        topic_embedding_files: If None, use default (title and description). Otherwise should be "topics_neighbors_xxx.npy", for example topics_neighbors_isomap_4.npy.
        """
        assert batch_size % 2 == 0, "Batch size must be even"

        contents_embeddings = np.concatenate([np.load("generated_data/mininet_L12_english384/contents_title.npy"),
                                              np.load(
                                                  "generated_data/mininet_L12_english384/contents_description.npy")], axis=1)
        if topic_embedding_file is None:
            topics_embeddings = np.concatenate([np.load("generated_data/mininet_L12_english384/topics_title.npy"),
                                                np.load(
                                                    "generated_data/mininet_L12_english384/topics_description.npy")], axis=1)
        else:
            topics_embeddings = np.load(os.path.join("generated_data/mininet_L12_english384", topic_embedding_file))

        # Convert contents_embeddings and topics_embeddings to torch tensors
        self.contents_embeddings = torch.tensor(contents_embeddings, dtype=torch.float32, device=device)
        self.topics_embeddings = torch.tensor(topics_embeddings, dtype=torch.float32, device=device)
        del contents_embeddings
        del topics_embeddings

        self.classifier = CSIC_classifier.Classifier(input_gaussian_noise=input_gaussian_noise,
                                                hidden_dropout_rate=dropout_rate).to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.loss_fn = torch.nn.BCELoss()

        self.epoch = 0
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

    def obtain_metrics(self, use_test_set=False):
        with torch.no_grad():
            if use_test_set:
                topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(self.test_batch_size // 2, self.test_batch_size // 2)
            else:
                topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_train_sample(self.test_batch_size // 2, self.test_batch_size // 2)
            while np.sum(cor) == 0:
                if use_test_set:
                    topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_test_sample(self.test_batch_size // 2,
                                                                                            self.test_batch_size // 2)
                else:
                    topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_train_sample(self.test_batch_size // 2,
                                                                                             self.test_batch_size // 2)
            contents_embeddings_batch = self.contents_embeddings[contents_num_id, :]
            topics_embeddings_batch = self.topics_embeddings[topics_num_id, :]
            labels_batch = torch.tensor(cor, dtype=torch.float32, device=device)

            outputs = torch.squeeze(self.classifier(torch.cat([contents_embeddings_batch, topics_embeddings_batch], dim=1)),
                                    dim=1)
            loss = self.loss_fn(outputs, labels_batch)

            actual_labels = cor.astype(dtype=np.int32)
            predicted_labels = outputs.cpu().numpy() > 0.5

            # Compute accuracy, precision, recall. Actual labels and predicted labels are integers either 0 or 1.
            accuracy = np.sum(actual_labels == predicted_labels) / len(actual_labels)
            if np.sum(predicted_labels == 1) == 0:
                precision = 0.0
            else:
                precision = np.sum((actual_labels == 1) & (predicted_labels == 1)) / np.sum(predicted_labels == 1)
            recall = np.sum((actual_labels == 1) & (predicted_labels == 1)) / np.sum(actual_labels == 1)
        return loss.item(), accuracy, precision, recall

    def run_one_epoch(self, n_steps):
        for k in range(n_steps):
            topics_num_id, contents_num_id, cor = CSIC_data_bert.obtain_train_sample(self.batch_size // 2, self.batch_size // 2)
            contents_embeddings_batch = self.contents_embeddings[contents_num_id, :]
            topics_embeddings_batch = self.topics_embeddings[topics_num_id, :]

            labels_batch = torch.tensor(cor, dtype=torch.float32, device=device)

            self.optimizer.zero_grad()
            outputs = torch.squeeze(self.classifier(torch.cat([contents_embeddings_batch, topics_embeddings_batch], dim=1)),
                                    dim=1)
            loss = self.loss_fn(outputs, labels_batch)
            loss.backward()
            self.optimizer.step()

        train_loss, train_accuracy, train_precision, train_recall = self.obtain_metrics(use_test_set=False)
        test_loss, test_accuracy, test_precision, test_recall = self.obtain_metrics(use_test_set=True)
        return train_loss, train_accuracy, train_precision, train_recall, test_loss, test_accuracy, test_precision, test_recall


def train_classifier(input_gaussian_noise, batch_size=2048, test_batch_size=32768, lr=0.0001, dropout_rate=0.1, topic_embedding_file=None, save_path="saved_models/CSIC_classifier.pt"):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10.8, 10.8))
    classifier = ClassifierTraining(input_gaussian_noise=input_gaussian_noise, batch_size=batch_size, test_batch_size=test_batch_size, lr=lr, dropout_rate=dropout_rate, topic_embedding_file=topic_embedding_file, save_path=save_path)

    train_loss = []
    train_accuracy = []
    train_precision = []
    train_recall = []
    test_loss = []
    test_accuracy = []
    test_precision = []
    test_recall = []

    def update(frame):
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                axs[i, j].clear()

        latest_train_loss, latest_train_accuracy, latest_train_precision, latest_train_recall,\
            latest_test_loss, latest_test_accuracy, latest_test_precision, latest_test_recall = classifier.run_one_epoch(100)
        train_loss.append(latest_train_loss)
        train_accuracy.append(latest_train_accuracy)
        train_precision.append(latest_train_precision)
        train_recall.append(latest_train_recall)
        test_loss.append(latest_test_loss)
        test_accuracy.append(latest_test_accuracy)
        test_precision.append(latest_test_precision)
        test_recall.append(latest_test_recall)

        axs[0, 0].plot(train_loss, label="Train")
        axs[0, 0].plot(test_loss, label="Test")
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend()

        axs[0, 1].plot(train_accuracy, label="Train")
        axs[0, 1].plot(test_accuracy, label="Test")
        axs[0, 1].legend()
        axs[0, 1].set_title("Accuracy")

        axs[1, 0].plot(train_precision, label="Train")
        axs[1, 0].plot(test_precision, label="Test")
        axs[1, 0].legend()
        axs[1, 0].set_title("Precision")

        axs[1, 1].plot(train_recall, label="Train")
        axs[1, 1].plot(test_recall, label="Test")
        axs[1, 1].legend()
        axs[1, 1].set_title("Recall")
        return axs


    anim = FuncAnimation(fig, update, frames=50, interval=50)

    # Show the animation
    plt.show()

train_classifier(input_gaussian_noise=0.1, batch_size=2048, lr=0.01, dropout_rate=0.1, topic_embedding_file=None, save_path="saved_models/CSIC_classifier.pt")
# print(device)