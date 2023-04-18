import tensorflow as tf

class FullyConnectedSubmodel(tf.keras.Model):
    def __init__(self, units_size = 512):
        super(FullyConnectedSubmodel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name="fc_dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name="fc_dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name="fc_dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name="fc_dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.3)
        self.dense5 = tf.keras.layers.Dense(units=units_size, activation="relu", name="fc_dense5")
        self.dropout5 = tf.keras.layers.Dropout(rate=0.3)

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False):
        t = data
        self.dropout1(self.dense1(t), training=training)
        self.dropout2(self.dense2(t), training=training)
        self.dropout3(self.dense3(t), training=training)
        self.dropout4(self.dense4(t), training=training)
        return self.dropout5(self.dense5(t), training=training)

def concat_siamese_data(data1, data2):
    return {"left": tf.concat([data1["left"], data2["left"]], axis=-1), "right": tf.concat([data1["right"], data2["right"]], axis=-1)}

class SiameseTwinSubmodel(tf.keras.Model):
    def __init__(self, units_size=512):
        super(SiameseTwinSubmodel, self).__init__()
        self.left_model = FullyConnectedSubmodel(units_size=units_size // 2)
        self.right_model = FullyConnectedSubmodel(units_size=units_size // 2)

    def call(self, data, training=False):
        return {"left": self.left_model(data["left"]), "right": self.left_model(data["right"])}