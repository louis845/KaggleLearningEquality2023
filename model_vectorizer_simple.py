# the training steps are commented out since it requires quite a lot of GPU memory!!

import tensorflow as tf
import tensorflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_vectorizer
# import data_vectorizer_tf

class Model(tf.keras.Model):
    def __init__(self, zero_to_one_ratio, training_initial_sample_size=1000, steps_per_epoch=50):
        super(Model, self).__init__()

        self.zero_to_one_ratio = zero_to_one_ratio
        self.training_initial_sample_size = training_initial_sample_size
        self.steps_per_epoch = steps_per_epoch

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
        self.full_accuracy = tf.keras.metrics.BinaryAccuracy(name="full_accuracy", threshold=0.7)
        self.full_precision = tf.keras.metrics.Precision(name="full_precision", thresholds=0.7)
        self.full_recall = tf.keras.metrics.Recall(name="full_recall", thresholds=0.7)
        self.test_precision = tf.keras.metrics.Precision(name="test_precision", thresholds=0.7)
        self.test_recall = tf.keras.metrics.Recall(name="test_recall", thresholds=0.7)

        self.num_step = 0

    def compile(self):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)

    def call(self, mdata):
        content_title, content_description, topic_title, topic_description = mdata
        t1 = tf.math.reduce_sum(self.content_title_embedding(content_title), axis=1)
        t2 = tf.math.reduce_sum(self.content_description_embedding(content_description), axis=1)
        t3 = tf.math.reduce_sum(self.topic_title_embedding(topic_title), axis=1)
        t4 = tf.math.reduce_sum(self.topic_description_embedding(topic_description), axis=1)

        embedding_result = self.concat_layer([t1, t2, t3, t4])
        return self.dense3(self.dense2(self.dense1(self.relu1(embedding_result))))

    """def obtain_tensor_values(self, initial_sample_size=1000, zero_to_one_ratio=None):
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

    def obtain_tensor_values2(self, initial_sample_size=1000, zero_to_one_ratio=None):
        contents_list, topics_list, correlations = data_vectorizer_tf.random_train_batch_sample(
            initial_sample_size=initial_sample_size, zero_to_one_ratio=zero_to_one_ratio)

        contents_strings = data_vectorizer_tf.obtain_train_contents_vector(contents_list)
        topics_strings = data_vectorizer_tf.obtain_train_topics_vector(topics_list)

        content_title, content_description = contents_strings
        topic_title, topic_description = topics_strings

        y = tf.constant(correlations)

        return content_title, content_description, topic_title, topic_description, y

    def obtain_test_values(self, initial_sample_size=1000, zero_to_one_ratio=None):
        contents_list, topics_list, correlations = data_vectorizer_tf.random_test_batch_sample(
            initial_sample_size=initial_sample_size, zero_to_one_ratio=zero_to_one_ratio)

        contents_strings = data_vectorizer_tf.obtain_train_contents_vector(contents_list)
        topics_strings = data_vectorizer_tf.obtain_train_topics_vector(topics_list)

        content_title, content_description = contents_strings
        topic_title, topic_description = topics_strings

        y = tf.constant(correlations)

        return content_title, content_description, topic_title, topic_description, y

    def train_step(self, data):
        # feedforward + backpropragation with training set
        if self.zero_to_one_ratio < 100:
            content_title, content_description, topic_title, topic_description, y = self.obtain_tensor_values(
                self.training_initial_sample_size, self.zero_to_one_ratio)
        else:
            content_title, content_description, topic_title, topic_description, y = self.obtain_tensor_values2(
                self.training_initial_sample_size, self.zero_to_one_ratio)

        with tf.GradientTape() as tape:
            y_pred = self((content_title, content_description, topic_title, topic_description))
            loss = self.loss(y, y_pred)

        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for m in self.metrics:
            m.update_state(y, y_pred)

        # feedforward with test set for metrics
        if self.num_step % self.steps_per_epoch == 0:
            content_title, content_description, topic_title, topic_description, y = self.obtain_tensor_values2(600)

            y_pred = self((content_title, content_description, topic_title, topic_description))

            for m in self.full_metrics:
                m.update_state(y, y_pred)

            content_title, content_description, topic_title, topic_description, y = self.obtain_test_values(600)

            y_pred = self((content_title, content_description, topic_title, topic_description))

            for m in self.test_metrics:
                m.update_state(y, y_pred)
        self.num_step += 1
        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics}, **{m.name: m.result() for m in self.full_metrics},
                **{m.name: m.result() for m in self.test_metrics}}"""

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]

    @property
    def test_metrics(self):
        return [self.test_precision, self.test_recall]

    @property
    def full_metrics(self):
        return [self.full_accuracy, self.full_precision, self.full_recall]