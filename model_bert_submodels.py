import tensorflow as tf

class FullyConnectedSubmodel(tf.keras.Model):
    def __init__(self, units_size = 512, final_layer_size = None, name = "fully_con"):
        super(FullyConnectedSubmodel, self).__init__(name = name)
        if final_layer_size is None:
            final_layer_size = units_size
        self.sname = name
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.3)
        self.dense5 = tf.keras.layers.Dense(units=final_layer_size, activation="relu", name=name + "_fc_dense5")
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
        self.save_weights(model_folder + "/" + str(epoch) + self.sname)

class FullyConnectedSmallSubmodel(tf.keras.Model):
    def __init__(self, units_size = 512, final_layer_size = None, name = "fully_con_small"):
        super(FullyConnectedSmallSubmodel, self).__init__(name = name)
        if final_layer_size is None:
            final_layer_size = units_size
        self.sname = name
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name=name + "_fc_dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=final_layer_size, activation="relu", name=name + "_fc_dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False):
        t = data
        t = self.dropout1(self.dense1(t), training=training)
        return self.dropout2(self.dense2(t), training=training)

    def msave_weights(self, model_folder, epoch):
        self.save_weights(model_folder + "/" + str(epoch) + self.sname)
class FullyConnectedSubmodelMiniClass(tf.keras.Model):
    def __init__(self, units_size = 512, name = "fully_con_mini"):
        super(FullyConnectedSubmodelMiniClass, self).__init__(name = name)
        self.sname = name
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
        self.save_weights(model_folder + "/" + str(epoch) + self.sname)

def concat_siamese_data(data1, data2):
    return {"left": tf.concat([data1["left"], data2["left"]], axis=-1), "right": tf.concat([data1["right"], data2["right"]], axis=-1)}

def combine_siamese_data(data):
    return tf.concat([data["left"], data["right"]], axis=-1)

class SiameseTwinSubmodel(tf.keras.Model):
    def __init__(self, units_size=512, final_layer_size = None, name = "sm_model", left_name = "sm_left", right_name = "sm_right"):
        super(SiameseTwinSubmodel, self).__init__(name = name)
        self.left_model = FullyConnectedSubmodel(units_size=units_size, name = left_name, final_layer_size = final_layer_size)
        self.right_model = FullyConnectedSubmodel(units_size=units_size, name = right_name, final_layer_size = final_layer_size)

    def call(self, data, training=False):
        return {"left": self.left_model(data["left"], training=training),
                "right": self.right_model(data["right"], training=training)}

    def msave_weights(self, model_folder, epoch):
        self.left_model.msave_weights(model_folder, epoch)
        self.right_model.msave_weights(model_folder, epoch)

class SiameseTwinSmallSubmodel(tf.keras.Model):
    def __init__(self, units_size=512, final_layer_size = None, name = "sm_model", left_name = "sm_left", right_name = "sm_right"):
        super(SiameseTwinSmallSubmodel, self).__init__(name = name)
        self.left_model = FullyConnectedSmallSubmodel(units_size=units_size, name = left_name, final_layer_size = final_layer_size)
        self.right_model = FullyConnectedSmallSubmodel(units_size=units_size, name = right_name, final_layer_size = final_layer_size)

    def call(self, data, training=False):
        return {"left": self.left_model(data["left"], training=training),
                "right": self.right_model(data["right"], training=training)}

    def msave_weights(self, model_folder, epoch):
        self.left_model.msave_weights(model_folder, epoch)
        self.right_model.msave_weights(model_folder, epoch)


class StepupSubmodel(tf.keras.Model):
    # same as the stepup model, except that no noise for inputs, and no set reduction (axis1) on outputs.
    def __init__(self, units_size=512, name = "stepup_submodel"):
        super(StepupSubmodel, self).__init__(name = name)

        self.sname = name
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

    def msave_weights(self, model_folder, epoch):
        self.save_weights(model_folder + "/" + str(epoch) + self.sname)
    def call(self, data, training=False):
        # we assume CONTENTS is in the left side, and TOPICS is in the right side
        first_layer1 = tf.concat([data["first_layer1"]["left"], data["first_layer1"]["right"]], axis=-1)
        first_layer2 = tf.concat([data["first_layer2"]["left"], data["first_layer2"]["right"]], axis=-1)

        t = self.dropout1(self.dense1(first_layer1), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        res_dropout4 = self.dropout4(self.dense4(t), training=training)

        overshoot_fullresult = self.dropoutOvershoot(self.denseOvershoot(res_dropout4), training=training)
        overshoot_result = self.finalOvershoot(overshoot_fullresult)

        t = self.dropout1_fp(self.dense1_fp(tf.concat([
            first_layer2,
            overshoot_fullresult
        ], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3_fp(self.dense3_fp(t), training=training)
        t = self.dropout4_fp(self.dense4_fp(t), training=training)
        t = self.dropout5_fp(self.dense5_fp(t), training=training)
        t = self.final(t)

        return {"t": t, "overshoot_result":overshoot_result}

    def evaluate_full_res(self, data, training=False):
        first_layer1 = tf.concat([data["first_layer1"]["left"], data["first_layer1"]["right"]], axis=-1)
        first_layer2 = tf.concat([data["first_layer2"]["left"], data["first_layer2"]["right"]], axis=-1)

        t = self.dropout1(self.dense1(first_layer1), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        res_dropout4 = self.dropout4(self.dense4(t), training=training)

        overshoot_fullresult = self.dropoutOvershoot(self.denseOvershoot(res_dropout4), training=training)
        overshoot_result = self.finalOvershoot(overshoot_fullresult)

        t = self.dropout1_fp(self.dense1_fp(tf.concat([
            first_layer2,
            overshoot_fullresult
        ], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3_fp(self.dense3_fp(t), training=training)
        t = self.dropout4_fp(self.dense4_fp(t), training=training)
        t = self.dropout5_fp(self.dense5_fp(t), training=training)
        t = self.final(t)
        return tf.concat([tf.reduce_max(t, axis=1), tf.reduce_max(overshoot_result, axis=1)], axis=1)