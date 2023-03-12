import time

import data_bert
import data_bert_restriction
import data_bert_tree_struct
import model_bert_fix
import tensorflow as tf
import numpy as np
import config
import gc
import data_bert_sampler
import sys

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise_topics=0.05, init_noise_overshoot_topics=0.2, init_noise_dampen_topics=0.2,
                 init_noise_contents=0.05, init_noise_overshoot_contents=0.2, init_noise_dampen_contents=0.2,
                 init_noise_lang=0.2, init_noise_overshoot_lang=0.3, init_noise_dampen_lang=0.3):
        super(Model, self).__init__()

        self.units_size = units_size

        self.training_sampler = None
        self.training_max_size = None

        # standard stuff
        self.dropout0_topics_dampen = tf.keras.layers.GaussianNoise(stddev=init_noise_dampen_topics)
        self.dropout0_topics_overshoot = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_topics)
        self.dropout0_feed_topics = tf.keras.layers.GaussianNoise(stddev=init_noise_topics)

        self.dropout0_contents_dampen = tf.keras.layers.GaussianNoise(stddev=init_noise_dampen_contents)
        self.dropout0_contents_overshoot = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_contents)
        self.dropout0_feed_contents = tf.keras.layers.GaussianNoise(stddev=init_noise_contents)

        self.dropout0_lang_dampen = tf.keras.layers.GaussianNoise(stddev=init_noise_dampen_lang)
        self.dropout0_lang_overshoot = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_lang)
        self.dropout0_feed_lang = tf.keras.layers.GaussianNoise(stddev=init_noise_lang)

        self.dense1_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_os")
        self.dropout1_os = tf.keras.layers.Dropout(rate=0.3)
        self.dense2_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_os")
        self.dropout2_os = tf.keras.layers.Dropout(rate=0.3)
        self.dense3_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_os")
        self.dropout3_os = tf.keras.layers.Dropout(rate=0.3)
        self.dense4_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_os")
        self.dropout4_os = tf.keras.layers.Dropout(rate=0.3)
        self.dense5_os = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_os")
        self.dropout5_os = tf.keras.layers.Dropout(rate=0.3)

        self.dense1_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1_dp")
        self.dropout1_dp = tf.keras.layers.Dropout(rate=0.3)
        self.dense2_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2_dp")
        self.dropout2_dp = tf.keras.layers.Dropout(rate=0.3)
        self.dense3_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3_dp")
        self.dropout3_dp = tf.keras.layers.Dropout(rate=0.3)
        self.dense4_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4_dp")
        self.dropout4_dp = tf.keras.layers.Dropout(rate=0.3)
        self.dense5_dp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_dp")
        self.dropout5_dp = tf.keras.layers.Dropout(rate=0.3)

        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid", name="finalOvershoot")
        self.finalDampen = tf.keras.layers.Dense(units=1, activation="sigmoid", name="finalOvershoot")

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

        # loss functions and eval metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision = tf.keras.metrics.Precision(name="precision")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.entropy = tf.keras.metrics.BinaryCrossentropy(name="entropy")
        self.entropy_large_set = tf.keras.metrics.BinaryCrossentropy(name="entropy_large_set")

        # metrics for test set
        self.custom_metrics = None
        self.custom_stopping_func = None

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False, actual_y=None, final_tree_level=None):
        contents_description = tf.squeeze(data["contents"]["description"], axis=1)
        contents_title = tf.squeeze(data["contents"]["title"], axis=1)
        contents_lang = tf.squeeze(data["contents"]["lang"], axis=1)
        topics_description = tf.squeeze(data["topics"]["description"], axis=1)
        topics_title = tf.squeeze(data["topics"]["title"], axis=1)
        topics_lang = tf.squeeze(data["topics"]["lang"], axis=1)
        # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))

        contents_text_info = tf.concat([contents_description, contents_title], axis=-1)
        topics_text_info = tf.concat([topics_description, topics_title], axis=-1)

        shape = contents_description.shape

        first_layer_overshoot_contents = self.dropout0_contents_overshoot(contents_text_info, training=training)
        first_layer_overshoot_topics = self.dropout0_topics_overshoot(topics_text_info, training=training)
        first_layer_overshoot_contents_lang = self.dropout0_lang_overshoot(contents_lang, training=training)
        first_layer_overshoot_topics_lang = self.dropout0_lang_overshoot(topics_lang, training=training)

        first_layer_dampen_contents = self.dropout0_contents_dampen(contents_text_info, training=training)
        first_layer_dampen_topics = self.dropout0_topics_dampen(topics_text_info, training=training)
        first_layer_dampen_contents_lang = self.dropout0_lang_dampen(contents_lang, training=training)
        first_layer_dampen_topics_lang = self.dropout0_lang_dampen(topics_lang, training=training)

        first_layer2_contents = self.dropout0_feed_contents(contents_text_info, training=training)
        first_layer2_topics = self.dropout0_feed_topics(topics_text_info, training=training)
        first_layer2_contents_lang = self.dropout0_feed_lang(contents_lang, training=training)
        first_layer2_topics_lang = self.dropout0_feed_lang(topics_lang, training=training)

        first_layer_overshoot = tf.concat(
            [first_layer_overshoot_contents, first_layer_overshoot_contents_lang, first_layer_overshoot_topics, first_layer_overshoot_topics_lang],
            axis=-1)
        first_layer_dampen = tf.concat(
            [first_layer_dampen_contents, first_layer_dampen_contents_lang, first_layer_dampen_topics, first_layer_dampen_topics_lang],
            axis=-1)
        first_layer2 = tf.concat(
            [first_layer2_contents, first_layer2_contents_lang, first_layer2_topics, first_layer2_topics_lang],
            axis=-1)

        t = self.dropout1_os(self.dense1_os(first_layer_overshoot), training=training)
        t = self.dropout2_os(self.dense2_os(t), training=training)
        t = self.dropout3_os(self.dense3_os(t), training=training)
        t = self.dropout4_os(self.dense4_os(t), training=training)
        overshoot_full_result = self.dropout5_os(self.dense5_os(t), training=training)
        overshoot_result = self.finalOvershoot(overshoot_full_result)

        t = self.dropout1_dp(self.dense1_dp(first_layer_dampen), training=training)
        t = self.dropout2_dp(self.dense2_dp(t), training=training)
        t = self.dropout3_dp(self.dense3_dp(t), training=training)
        t = self.dropout4_dp(self.dense4_dp(t), training=training)
        dampen_full_result = self.dropout5_dp(self.dense5_dp(t), training=training)
        dampen_result = self.finalDampen(dampen_full_result)

        t = self.dropout1_fp(self.dense1_fp(tf.concat([
            first_layer2, overshoot_full_result, dampen_full_result
        ], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3_fp(self.dense3_fp(t), training=training)
        t = self.dropout4_fp(self.dense4_fp(t), training=training)
        t = self.dropout5_fp(self.dense5_fp(t), training=training)
        t = self.final(t)

        return tf.concat([t, overshoot_result, dampen_result], axis=1)

    def compile(self, weight_decay=0.01, learning_rate=0.0005):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000
        self.prev_entropy = None

    def set_sampling_functions_with_pack(self, rspack: data_bert_restriction.RestrictionSamplerPack):
        # saves the functions that generates whether a tuple (topic, content) is correlated.
        self.tuple_choice_sampler = data_bert_sampler.default_sampler_instance

        # generates the overshoot functions
        self.tuple_choice_sampler_overshoot = data_bert_sampler.default_sampler_overshoot2_instance

        # generates the dampening functions
        self.tuple_choice_sampler_dampen = rspack.default_dampening_sampler_instance
        self.tuple_choice_sampler_dampen_overshoot = rspack.default_dampening_sampler_overshoot2_instance

    def set_sampling_functions_with_pack_interlaced(self, rspack: data_bert_restriction.RestrictionSamplerPack,
                                                    rspack_overshoot: data_bert_restriction.RestrictionSamplerPack):
        # saves the functions that generates whether a tuple (topic, content) is correlated.
        self.tuple_choice_sampler = data_bert_sampler.default_sampler_instance

        # generates the overshoot functions
        self.tuple_choice_sampler_overshoot = data_bert_sampler.default_sampler_overshoot2_instance

        # generates the dampening functions
        self.tuple_choice_sampler_dampen = rspack.default_dampening_sampler_instance
        self.tuple_choice_sampler_dampen_overshoot = rspack_overshoot.default_dampening_sampler_overshoot2_instance

    def set_training_params(self, training_sample_size=15000, training_max_size=None,
                            training_sampler=None, custom_metrics=None, custom_stopping_func=None,
                            custom_tuple_choice_sampler=None, custom_tuple_choice_sampler_overshoot=None,
                            custom_tuple_choice_sampler_dampen=None):
        self.training_sample_size = training_sample_size
        self.training_max_size = training_max_size
        if training_sampler is not None:
            self.training_sampler = training_sampler
            s = self(self.training_sampler.obtain_input_data(np.array([0]), np.array([0])))
            del s
        if custom_metrics is not None:
            custom_metrics.set_training_sampler(self.training_sampler)
            self.custom_metrics = custom_metrics
        if custom_stopping_func is not None:
            self.custom_stopping_func = custom_stopping_func
        if custom_tuple_choice_sampler is not None:
            self.tuple_choice_sampler = custom_tuple_choice_sampler
        if custom_tuple_choice_sampler_overshoot is not None:
            self.tuple_choice_sampler_overshoot = custom_tuple_choice_sampler_overshoot
        if custom_tuple_choice_sampler_dampen is not None:
            self.tuple_choice_sampler_dampen = custom_tuple_choice_sampler_dampen

    @tf.function
    def call_optim_step(self, input_data, y, length1, length2, length3):
        with tf.GradientTape() as tape:
            y_pred = self(input_data, training=True)
            loss = self.loss(y, tf.concat([y_pred[:length1, 0], y_pred[length1:length2, 1],
                                           y_pred[length2:length3, 2],
                                           y_pred[length3:(length3 + length1), 0],
                                           y_pred[(length3 + length1):(length3 + length2), 1],
                                           y_pred[(length3 + length2):, 2]], axis=0))
        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    @tf.function
    def call_self(self, input_data):
        return self(input_data)

    def train_step(self, data):
        if self.tuple_choice_sampler.is_tree_sampler():
            return self.train_step_tree()
        for k in range(50):
            ratio1 = 1.0 / 15
            ratio2 = 7.0 / 15
            ratio3 = 7.0 / 15

            topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(
                int(0 * ratio1 * self.training_sample_size) // 8)
            topics_, contents_, cors_, class_ids_ = self.tuple_choice_sampler_dampen.obtain_train_sample(
                int(8 * ratio1 * self.training_sample_size) // 8)
            topics2, contents2, cors2, class_ids2 = self.tuple_choice_sampler_overshoot.obtain_train_sample(
                int(ratio2 * self.training_sample_size))
            topics3, contents3, cors3, class_ids3 = self.tuple_choice_sampler_dampen_overshoot.obtain_train_sample(
                int(ratio3 * self.training_sample_size))

            del cors, cors2, cors3, cors_, class_ids, class_ids_, class_ids2, class_ids3

            topics = np.concatenate([topics, topics_])
            contents = np.concatenate([contents, contents_])

            y1 = self.tuple_choice_sampler.has_correlations(contents, topics, None)
            y2 = self.tuple_choice_sampler_overshoot.has_correlations(contents2, topics2, None)
            y3 = self.tuple_choice_sampler_dampen_overshoot.has_correlations(contents3, topics3, None)

            y0 = np.tile(np.concatenate([y1, y2, y3]), 2)
            y = tf.constant(y0)


            topics = np.concatenate([topics, topics2, topics3])
            contents = np.concatenate([contents, contents2, contents3])

            input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)

            ctime = time.time()
            self.call_optim_step(input_data, y, len(y1), len(y1) + len(y2), len(y1) + len(y2) + len(y3))
            ctime = time.time() - ctime

        if self.training_max_size is None:
            limit = 9223372036854775807
            limit_sq = 9223372036854775807
        else:
            limit = self.training_max_size

        # evaluation at larger subset
        ratio1 = 1.0 / 15
        ratio2 = 7.0 / 15
        ratio3 = 7.0 / 15

        topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(
            int(0 * ratio1 * self.training_sample_size) // 8)
        topics_, contents_, cors_, class_ids_ = self.tuple_choice_sampler_dampen.obtain_train_sample(
            int(8 * ratio1 * self.training_sample_size) // 8)
        topics2, contents2, cors2, class_ids2 = self.tuple_choice_sampler_overshoot.obtain_train_sample(
            int(ratio2 * self.training_sample_size))
        topics3, contents3, cors3, class_ids3 = self.tuple_choice_sampler_dampen_overshoot.obtain_train_sample(
            int(ratio3 * self.training_sample_size))

        del cors, cors2, cors3, cors_, class_ids, class_ids_, class_ids2, class_ids3

        topics = np.concatenate([topics, topics_])
        contents = np.concatenate([contents, contents_])

        y1 = self.tuple_choice_sampler.has_correlations(contents, topics, None)
        y2 = self.tuple_choice_sampler_overshoot.has_correlations(contents2, topics2, None)
        y3 = self.tuple_choice_sampler_dampen_overshoot.has_correlations(contents3, topics3, None)

        y0 = np.tile(np.concatenate([y1, y2, y3]), 2)
        y = tf.constant(y0)

        topics = np.concatenate([topics, topics2, topics3])
        contents = np.concatenate([contents, contents2, contents3])

        input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)

        y_pred = self.call_self(input_data)
        length1, length2, length3 = len(y1), len(y1) + len(y2), len(y1) + len(y2) + len(y3)
        y_pred = tf.concat([y_pred[:length1, 0], y_pred[length1:length2, 1],
                                           y_pred[length2:length3, 2],
                                           y_pred[length3:(length3 + length1), 0],
                                           y_pred[(length3 + length1):(length3 + length2), 1],
                                           y_pred[(length3 + length2):, 2]], axis=0)
        self.entropy_large_set.update_state(y, y_pred)

        new_entropy = self.entropy_large_set.result()
        if (self.prev_entropy is not None) and new_entropy > self.prev_entropy * 1.05:
            print(
                "---------------------------------WARNING: Training problem: entropy has increased! ---------------------------------")
        else:
            self.prev_entropy = new_entropy

        for m in self.metrics:
            m.update_state(y, y_pred)

        # eval other test metrics
        if self.custom_metrics is not None:
            self.custom_metrics.update_metrics(self, limit)

        # early stopping
        if (self.custom_stopping_func is not None) and self.custom_stopping_func.evaluate(self.custom_metrics, self):
            self.stop_training = True

        return {**{m.name: m.result() for m in self.metrics},
                "entropy_large_set": self.entropy_large_set.result(), **self.custom_metrics.obtain_metrics()}

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]

class DynamicMetrics(model_bert_fix.CustomMetrics):

    TRAIN = 1
    TRAIN_SQUARE = 2
    TEST = 3
    TEST_SQUARE = 4
    TRAIN_OVERSHOOT = 5
    TRAIN_SQUARE_OVERSHOOT = 6
    TEST_OVERSHOOT = 7
    TEST_SQUARE_OVERSHOOT = 8
    TRAIN_DAMPEN = 9
    TRAIN_SQUARE_DAMPEN = 10
    TEST_DAMPEN = 11
    TEST_SQUARE_DAMPEN = 12

    def __init__(self):
        model_bert_fix.CustomMetrics.__init__(self)
        self.metrics = [] # a lists of dicts, containing the metrics, and the data_bert_sampler.SamplerBase which contains the metric
        self.tree_metrics = []

    def add_metric(self, name, tuple_choice_sampler, sample_choice = TEST, threshold = 0.5):
        accuracy = tf.keras.metrics.BinaryAccuracy(name = name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name = name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name = name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name = name + "_entropy")

        accuracy_nolang = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy_nolang", threshold=threshold)
        precision_nolang = tf.keras.metrics.Precision(name=name + "_precision_nolang", thresholds=threshold)
        recall_nolang = tf.keras.metrics.Recall(name=name + "_recall_nolang", thresholds=threshold)
        entropy_nolang = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy_nolang")
        self.metrics.append({"metrics": [accuracy, precision, recall, entropy, accuracy_nolang, precision_nolang, recall_nolang, entropy_nolang], "sampler": tuple_choice_sampler, "sample_choice": sample_choice})

    def update_metrics(self, model, sample_size_limit):
        for k in range(len(self.metrics)):
            kmetrics = self.metrics[k]["metrics"]
            sampler = self.metrics[k]["sampler"]
            sample_choice = self.metrics[k]["sample_choice"]

            if (sample_choice == DynamicMetrics.TRAIN or sample_choice == DynamicMetrics.TRAIN_OVERSHOOT or sample_choice == DynamicMetrics.TRAIN_DAMPEN):
                topics, contents, cors, class_id = sampler.obtain_train_sample(min(60000, sample_size_limit))
            elif (sample_choice == DynamicMetrics.TRAIN_SQUARE or sample_choice == DynamicMetrics.TRAIN_SQUARE_OVERSHOOT or sample_choice == DynamicMetrics.TRAIN_SQUARE_DAMPEN):
                topics, contents, cors, class_id = sampler.obtain_train_square_sample(min(360000, sample_size_limit))
            elif (sample_choice == DynamicMetrics.TEST or sample_choice == DynamicMetrics.TEST_OVERSHOOT or sample_choice == DynamicMetrics.TEST_DAMPEN):
                topics, contents, cors, class_id = sampler.obtain_test_sample(min(60000, sample_size_limit))
            elif (sample_choice == DynamicMetrics.TEST_SQUARE or sample_choice == DynamicMetrics.TEST_SQUARE_OVERSHOOT or sample_choice == DynamicMetrics.TEST_SQUARE_DAMPEN):
                topics, contents, cors, class_id = sampler.obtain_test_square_sample(min(360000, sample_size_limit))

            y = tf.constant(cors, dtype=tf.float32)

            input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
            if sample_choice <= 4:
                y_pred = model.call_self(input_data)[:, 0]
            elif sample_choice <= 8:
                y_pred = model.call_self(input_data)[:, 1]
            else:
                y_pred = model.call_self(input_data)[:, 2]
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
            if sample_choice <= 4:
                y_pred = model.call_self(input_data)[:, 0]
            elif sample_choice <= 8:
                y_pred = model.call_self(input_data)[:, 1]
            else:
                y_pred = model.call_self(input_data)[:, 2]

            for j in range(4,8):
                kmetrics[j].update_state(y, y_pred)
            gc.collect()
    def obtain_metrics(self):
        metrics_list = [metr for met in self.metrics for metr in met["metrics"]] + [metr for met in self.tree_metrics for metr in met["metrics"]]
        return {m.name: m.result() for m in metrics_list}

    def get_test_entropy_metric(self):
        for k in range(len(self.metrics)):
            mmetrics = self.metrics[k]["metrics"]
            if mmetrics[3].name == "test_entropy":
                return mmetrics[3].result()
        raise Exception("No metrics found!")

    def get_test_overshoot_entropy_metric(self):
        for k in range(len(self.metrics)):
            mmetrics = self.metrics[k]["metrics"]
            if mmetrics[3].name == "test_overshoot_entropy":
                return mmetrics[3].result()
        raise Exception("No metrics found!")


default_metrics = None

def generate_default_metrics(rspack: data_bert_restriction.RestrictionSamplerPack):
    global default_metrics
    default_metrics = DynamicMetrics()
    default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST)
    default_metrics.add_metric("test_final_dampen", rspack.default_dampening_sampler_instance, sample_choice = DynamicMetrics.TEST)
    default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST_SQUARE)
    default_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
    default_metrics.add_metric("test_dampen", rspack.default_dampening_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_DAMPEN)
    default_metrics.add_metric("test_dampen_usual", data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_DAMPEN)
    default_metrics.add_metric("test_square_dampen", rspack.default_dampening_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE_DAMPEN)