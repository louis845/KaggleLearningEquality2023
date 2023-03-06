import tensorflow as tf

class FullyConnectedSubmodel(tf.keras.Model):
    def __init__(self, units_size = 512, name = "fully_con"):
        super(FullyConnectedSubmodel, self).__init__(name = name)
        self.sname = name
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.3)
        self.dense5 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense5")
        self.dropout5 = tf.keras.layers.Dropout(rate=0.3)

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False):
        t = data
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dropout4(self.dense4(t), training=training)
        return self.dropout5(self.dense5(t), training=training)

    def msave_weights(self, model_folder, epoch):
        self.save_weights(model_folder + str(epoch) + self.sname)
class FullyConnectedSubmodelMiniClass(tf.keras.Model):
    def __init__(self, units_size = 512, name = "fully_con_mini"):
        super(FullyConnectedSubmodelMiniClass, self).__init__(name = name)
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense_final = tf.keras.layers.Dense(units=1, activation="sigmoid", name=name + "_fc_final")

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False):
        t = data
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        return self.dense_final(t)

    def msave_weights(self, model_folder, epoch):
        self.save_weights(model_folder + str(epoch) + self.sname)

def concat_siamese_data(data1, data2):
    return {"left": tf.concat([data1["left"], data2["left"]], axis=-1), "right": tf.concat([data1["right"], data2["right"]], axis=-1)}

def combine_siamese_data(data):
    return tf.concat([data["left"], data["right"]], axis=-1)

class SiameseTwinSubmodel(tf.keras.Model):
    def __init__(self, units_size=512, name = "sm_model", left_name = "sm_left", right_name = "sm_right"):
        super(SiameseTwinSubmodel, self).__init__(name = name)
        self.left_model = FullyConnectedSubmodel(units_size=units_size // 2, name = left_name)
        self.right_model = FullyConnectedSubmodel(units_size=units_size // 2, name = right_name)

    def call(self, data, training=False):
        return {"left": self.left_model(data["left"], training=training),
                "right": self.right_model(data["right"], training=training)}

    def msave_weights(self, model_folder, epoch):
        self.left_model.msave_weights(model_folder, epoch)
        self.right_model.msave_weights(model_folder, epoch)