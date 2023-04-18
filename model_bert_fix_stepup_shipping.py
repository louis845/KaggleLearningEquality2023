import tensorflow as tf


# for shipping, we removed all the other stuff, except for model definition

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise=0.05, init_noise_overshoot=0.2):
        super(Model, self).__init__()

        self.units_size = units_size

        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.3)
        # the results of dense2 will be plugged into this.
        self.denseOvershoot = tf.keras.layers.Dense(units=units_size, activation="relu", name="denseOvershoot")
        self.dropoutOvershoot = tf.keras.layers.Dropout(rate=0.3)
        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid", name="finalOvershoot")

        # self.dropoutCombine1 = tf.keras.layers.Dropout(rate=0.5)
        # self.dropoutCombine2 = tf.keras.layers.Dropout(rate=0.5)

        # dense1_fp takes in the combined input of dense0 and denseOvershoot
        self.dense1_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_fp")
        self.dropout1_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense2_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_fp")
        self.dropout2_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense3_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_fp")
        self.dropout3_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense4_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_fp")
        self.dropout4_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense5_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_fp")
        self.dropout5_fp = tf.keras.layers.Dropout(rate=0.3)
        self.final = tf.keras.layers.Dense(units=1, activation="sigmoid", name="final")
        
        self.dense1_left_matrix = None
        self.dense1_right_matrix = None
        self.dense1_fp_left_matrix = None
        self.dense1_fp_right_matrix = None
        self.dense1_fp_residual_matrix = None

    def call(self, data, training=False):
        t = self.dropout1(self.dense1(data), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        res_dropout4 = self.dropout4(self.dense4(t), training=training)

        overshoot_fullresult = self.dropoutOvershoot(self.denseOvershoot(res_dropout4), training=training)
        overshoot_result = self.finalOvershoot(overshoot_fullresult)

        t = self.dropout1_fp(self.dense1_fp(tf.concat([
            data,
            overshoot_fullresult
        ], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3_fp(self.dense3_fp(t), training=training)
        t = self.dropout4_fp(self.dense4_fp(t), training=training)
        t = self.dropout5_fp(self.dense5_fp(t), training=training)
        t = self.final(t)
        return tf.concat([tf.reduce_max(t, axis=1), tf.reduce_max(overshoot_result, axis=1)], axis=1)

    def call_nonset_final_only(self, data):
        t = self.dense1(data)
        t = self.dense2(t)
        t = self.dense3(t)
        res_dropout4 = self.dense4(t)
        overshoot_fullresult = self.denseOvershoot(res_dropout4)

        t = self.dense1_fp(tf.concat([
            data,
            overshoot_fullresult
        ], axis=-1))
        t = self.dense2_fp(t)
        t = self.dense3_fp(t)
        t = self.dense4_fp(t)
        t = self.dense5_fp(t)
        t = self.final(t)
        return tf.squeeze(t, axis=1)

    def generate_first_layer_decompositions(self):
        assert self.units_size % 2 == 0
        assert self.dense1.weights[0].shape[0] % 2 == 0
        inputs_size = self.dense1.weights[0].shape[0] // 2
        self.dense1_left_matrix = self.dense1.weights[0][:inputs_size, :]
        self.dense1_right_matrix = self.dense1.weights[0][inputs_size:, :]
        self.dense1_fp_left_matrix = self.dense1_fp.weights[0][:inputs_size, :]
        self.dense1_fp_right_matrix = self.dense1_fp.weights[0][inputs_size:2 * inputs_size, :]
        self.dense1_fp_residual_matrix = self.dense1_fp.weights[0][2 * inputs_size:, :]

    def transform_left_data_dense1(self, data):
        return tf.matmul(data, self.dense1_left_matrix)
    def transform_right_data_dense1(self, data):
        return tf.matmul(data, self.dense1_right_matrix)

    def transform_left_data_dense1_fp(self, data):
        return tf.matmul(data, self.dense1_fp_left_matrix)
    def transform_right_data_dense1_fp(self, data):
        return tf.matmul(data, self.dense1_fp_right_matrix)

    def call_fast_dim_reduced(self, left_d1, right_d1, left_d1fp, right_d1fp):
        t = tf.nn.relu(tf.nn.bias_add(left_d1 + right_d1, self.dense1.weights[1]))
        t = self.dense2(t)
        t = self.dense3(t)
        res_dropout4 = self.dense4(t)
        overshoot_fullresult = self.denseOvershoot(res_dropout4)

        t = tf.nn.relu(tf.nn.bias_add(left_d1fp + right_d1fp +
                           tf.matmul(overshoot_fullresult, self.dense1_fp_residual_matrix),self.dense1_fp.weights[1]))
        t = self.dense2_fp(t)
        t = self.dense3_fp(t)
        t = self.dense4_fp(t)
        t = self.dense5_fp(t)
        t = self.final(t)
        return tf.squeeze(t, axis=1)