import time

import tensorflow as tf
import numpy as np
import gc

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512):
        super(Model, self).__init__()

        self.units_size = units_size

        self.dense1_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_os")
        self.dense2_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_os")
        self.dense3_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_os")
        self.dense4_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_os")
        self.dense5_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_os")

        self.dense1_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_dp")
        self.dense2_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_dp")
        self.dense3_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_dp")
        self.dense4_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_dp")
        self.dense5_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_dp")

        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid", name="finalOvershoot")
        self.finalDampen = tf.keras.layers.Dense(units=1, activation="sigmoid", name="finalOvershoot")

        # self.dropoutCombine1 = tf.keras.layers.Dropout(rate=0.5)
        # self.dropoutCombine2 = tf.keras.layers.Dropout(rate=0.5)

        # dense1_fp takes in the combined input of dense0 and denseOvershoot
        self.dense1_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_fp")
        self.dense2_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_fp")
        self.dense3_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_fp")
        self.dense4_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_fp")
        self.dense5_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_fp")
        self.final = tf.keras.layers.Dense(units=1, activation="sigmoid", name="final")

        self.dense1os_left_matrix = None
        self.dense1os_right_matrix = None
        self.dense1dp_left_matrix = None
        self.dense1dp_right_matrix = None
        self.dense1fp_left_matrix = None
        self.dense1fp_right_matrix = None
        self.dense1fp_residual_matrix = None

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data):
        first_layer_overshoot = data
        first_layer_dampen = data
        first_layer2 = data

        t = self.dense1_os(first_layer_overshoot)
        t = self.dense2_os(t)
        t = self.dense3_os(t)
        t = self.dense4_os(t)
        overshoot_full_result = self.dense5_os(t)
        overshoot_result = self.finalOvershoot(overshoot_full_result)

        t = self.dense1_dp(first_layer_dampen)
        t = self.dense2_dp(t)
        t = self.dense3_dp(t)
        t = self.dense4_dp(t)
        dampen_full_result = self.dense5_dp(t)
        dampen_result = self.finalDampen(dampen_full_result)

        t = self.dense1_fp(tf.concat([
            first_layer2, overshoot_full_result, dampen_full_result
        ], axis=-1))
        t = self.dense2_fp(t)
        t = self.dense3_fp(t)
        t = self.dense4_fp(t)
        t = self.dense5_fp(t)
        t = self.final(t)

        return tf.concat([t, overshoot_result, dampen_result], axis=1)

    def call_nonset_final_only(self, data):
        t = self.dense1_os(data)
        t = self.dense2_os(t)
        t = self.dense3_os(t)
        t = self.dense4_os(t)
        overshoot_full_result = self.dense5_os(t)

        t = self.dense1_dp(data)
        t = self.dense2_dp(t)
        t = self.dense3_dp(t)
        t = self.dense4_dp(t)
        dampen_full_result = self.dense5_dp(t)

        t = self.dense1_fp(tf.concat([
            data, overshoot_full_result, dampen_full_result
        ], axis=-1))
        t = self.dense2_fp(t)
        t = self.dense3_fp(t)
        t = self.dense4_fp(t)
        t = self.dense5_fp(t)
        t = self.final(t)

        return tf.squeeze(t, axis=1)

    def generate_first_layer_decompositions(self):
        assert self.units_size % 2 == 0
        assert self.dense1_os.weights[0].shape[0] % 2 == 0
        assert self.dense1_dp.weights[0].shape[0] % 2 == 0
        inputs_size = self.dense1_os.weights[0].shape[0] // 2

        self.dense1os_left_matrix = self.dense1_os.weights[0][:inputs_size, :]
        self.dense1os_right_matrix = self.dense1_os.weights[0][inputs_size:, :]

        self.dense1dp_left_matrix = self.dense1_dp.weights[0][:inputs_size, :]
        self.dense1dp_right_matrix = self.dense1_dp.weights[0][inputs_size:, :]

        self.dense1fp_left_matrix = self.dense1_fp.weights[0][:inputs_size, :]
        self.dense1fp_right_matrix = self.dense1_fp.weights[0][inputs_size:2 * inputs_size, :]
        self.dense1fp_residual_matrix = self.dense1_fp.weights[0][2 * inputs_size:, :]

    def transform_left_data_dense1os(self, data):
        return tf.matmul(data, self.dense1os_left_matrix)
    def transform_right_data_dense1os(self, data):
        return tf.matmul(data, self.dense1os_right_matrix)

    def transform_left_data_dense1dp(self, data):
        return tf.matmul(data, self.dense1dp_left_matrix)
    def transform_right_data_dense1dp(self, data):
        return tf.matmul(data, self.dense1dp_right_matrix)

    def transform_left_data_dense1fp(self, data):
        return tf.matmul(data, self.dense1fp_left_matrix)
    def transform_right_data_dense1fp(self, data):
        return tf.matmul(data, self.dense1fp_right_matrix)

    def call_fast_dim_reduced(self, left_d1os, right_d1os, left_d1dp, right_d1dp, left_d1fp, right_d1fp):
        t = tf.nn.relu(tf.nn.bias_add(left_d1os + right_d1os, self.dense1_os.weights[1]))
        t = self.dense2_os(t)
        t = self.dense3_os(t)
        t = self.dense4_os(t)
        overshoot_full_result = self.dense5_os(t)

        t = tf.nn.relu(tf.nn.bias_add(left_d1dp + right_d1dp, self.dense1_dp.weights[1]))
        t = self.dense2_dp(t)
        t = self.dense3_dp(t)
        t = self.dense4_dp(t)
        dampen_full_result = self.dense5_dp(t)

        t = tf.nn.relu(tf.nn.bias_add(left_d1fp + right_d1fp +
                                      tf.matmul(
                                          tf.concat([overshoot_full_result, dampen_full_result], axis=-1),
                                      self.dense1_fp_residual_matrix),
                                self.dense1_fp.weights[1]))
        t = self.dense2_fp(t)
        t = self.dense3_fp(t)
        t = self.dense4_fp(t)
        t = self.dense5_fp(t)
        t = self.final(t)

        return tf.squeeze(t, axis=1)