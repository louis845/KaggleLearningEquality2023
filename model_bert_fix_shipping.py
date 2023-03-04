import tensorflow as tf

# for shipping, we removed all the other stuff, except for model definition

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise=0.05):
        super(Model, self).__init__()

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis=2)

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianNoise(stddev=init_noise)
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.1)
        self.dense5 = tf.keras.layers.Dense(units=units_size // 4, activation="relu", name="dense5")
        self.dropout5 = tf.keras.layers.Dropout(rate=0.1)
        self.dense_final = tf.keras.layers.Dense(units=1, activation="sigmoid", name="dense_final")
    def call(self, data, training=False, actual_y=None, final_tree_level=None):
        embedding_result = data

        t = self.dropout0(embedding_result, training=training)
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dropout4(self.dense4(t), training=training)
        t = self.dropout5(self.dense5(t), training=training)
        t = self.dense_final(t)
        return tf.squeeze(tf.reduce_max(t, axis=1), axis=1)
