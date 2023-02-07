import sys
sys.path.insert(1, "/kaggle/input/learningequalityvectorizationml/")
import data
import data_vectorizer

def transform_string_to_list(x):
    if len(x) == 2:
        return []
    return [int(n) for n in x[1:-1].split(", ")]

data.contents["title_vectorize"] = data.contents["title_vectorize"].apply(transform_string_to_list)
data.contents["description_vectorize"] = data.contents["description_vectorize"].apply(transform_string_to_list)
data.topics["title_vectorize"] = data.topics["title_vectorize"].apply(transform_string_to_list)
data.topics["description_vectorize"] = data.topics["description_vectorize"].apply(transform_string_to_list)

import tensorflow as tf
import tensorflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelVectorizerSimple(tf.keras.Model):
    def __init__(self, zero_to_one_ratio):
        super(ModelVectorizerSimple, self).__init__()

        self.zero_to_one_ratio = zero_to_one_ratio

        # define layers here, input is (batch_size, string_sequence_length)
        # initial embedding layers. in below, we evaluate this with ragged tensors input (on second axis (1))
        word_dict_size = len(data_vectorizer.word_freqs_filtered)
        self.content_title_embedding = tf.keras.layers.Embedding(word_dict_size, 30)
        self.content_description_embedding = tf.keras.layers.Embedding(word_dict_size, 30)
        self.topic_title_embedding = tf.keras.layers.Embedding(word_dict_size, 30)
        self.topic_description_embedding = tf.keras.layers.Embedding(word_dict_size, 30)

        # concatenation layer
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)

        # standard stuff
        self.relu1 = tf.keras.layers.Activation('relu')
        self.dense1 = tf.keras.layers.Dense(units=30, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=30, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        # loss functions and eval metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision = tf.keras.metrics.Precision(name="precision")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.entropy = tf.keras.metrics.BinaryCrossentropy(name="entropy")

        # metrics for test set
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")
        self.test_precision = tf.keras.metrics.Precision(name="test_precision")
        self.test_recall = tf.keras.metrics.Recall(name="test_recall")
        self.test_entropy = tf.keras.metrics.BinaryCrossentropy(name="test_entropy")

    def compile(self):
        super(ModelVectorizerSimple, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, mdata):
        content_title, content_description, topic_title, topic_description = mdata
        t1 = tf.math.reduce_sum(self.content_title_embedding(content_title), axis=1)
        t2 = tf.math.reduce_sum(self.content_description_embedding(content_description), axis=1)
        t3 = tf.math.reduce_sum(self.topic_title_embedding(topic_title), axis=1)
        t4 = tf.math.reduce_sum(self.topic_description_embedding(topic_description), axis=1)

        embedding_result = self.concat_layer([t1, t2, t3, t4])
        return self.dense3(self.dense2(self.dense1(self.relu1(embedding_result))))

    def obtain_tensor_values(self, initial_sample_size=1000, zero_to_one_ratio=None):
        contents_list, topics_list, correlations = data_vectorizer.random_train_batch_sample(
            initial_sample_size=initial_sample_size, zero_to_one_ratio=zero_to_one_ratio)

        contents_strings = data_vectorizer.obtain_contents_vector(list(contents_list))
        topics_strings = data_vectorizer.obtain_topics_vector(list(topics_list))

        content_title = tf.ragged.constant(list(contents_strings["title_translate"]))
        content_description = tf.ragged.constant(list(contents_strings["description_translate"]))
        topic_title = tf.ragged.constant(list(topics_strings["title_translate"]))
        topic_description = tf.ragged.constant(list(topics_strings["description_translate"]))

        y = tf.constant(correlations)

        return content_title, content_description, topic_title, topic_description, y

    def train_step(self, data):
        # feedforward + backpropragation with training set
        content_title, content_description, topic_title, topic_description, y = self.obtain_tensor_values(1000,
                                                                                                          self.zero_to_one_ratio)

        with tf.GradientTape() as tape:
            y_pred = self((content_title, content_description, topic_title, topic_description))
            loss = self.loss(y, y_pred)

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for m in self.metrics:
            m.update_state(y, y_pred)

        # feedforward with test set for metrics
        content_title, content_description, topic_title, topic_description, y = self.obtain_tensor_values(400)

        y_pred = self((content_title, content_description, topic_title, topic_description))

        for m in self.test_metrics:
            m.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics}, **{m.name: m.result() for m in self.test_metrics}}

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]

    @property
    def test_metrics(self):
        return [self.test_accuracy, self.test_precision, self.test_recall, self.test_entropy]