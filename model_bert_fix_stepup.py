import data_bert
import data_bert_tree_struct
import model_bert_fix
import tensorflow as tf
import numpy as np
import config
import gc
import data_bert_sampler
#

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise_topics = 0.05, init_noise_overshoot_topics = 0.2, init_noise_contents = 0.05, init_noise_overshoot_contents = 0.2):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # standard stuff
        self.dropout0_topics = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_topics)
        self.dropout0_feed_topics = tf.keras.layers.GaussianNoise(stddev=init_noise_topics)
        self.dropout0_contents = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_contents)
        self.dropout0_feed_contents = tf.keras.layers.GaussianNoise(stddev=init_noise_contents)

        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.3)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.3)
        # the results of dense2 will be plugged into this.
        self.denseOvershoot = tf.keras.layers.Dense(units=units_size, activation="relu", name = "denseOvershoot")
        self.dropoutOvershoot = tf.keras.layers.Dropout(rate=0.3)
        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid", name = "finalOvershoot")

        # self.dropoutCombine1 = tf.keras.layers.Dropout(rate=0.5)
        # self.dropoutCombine2 = tf.keras.layers.Dropout(rate=0.5)

        # dense1_fp takes in the combined input of dense0 and denseOvershoot
        self.dense1_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense1_fp")
        self.dropout1_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense2_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense2_fp")
        self.dropout2_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense3_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense3_fp")
        self.dropout3_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense4_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense4_fp")
        self.dropout4_fp = tf.keras.layers.Dropout(rate=0.3)
        self.dense5_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense5_fp")
        self.dropout5_fp = tf.keras.layers.Dropout(rate=0.3)
        self.final = tf.keras.layers.Dense(units=1, activation="sigmoid", name = "final")

        # loss functions and eval metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision = tf.keras.metrics.Precision(name="precision")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.entropy = tf.keras.metrics.BinaryCrossentropy(name="entropy")
        self.entropy_large_set = tf.keras.metrics.BinaryCrossentropy(name="entropy_large_set")

        # metrics for test set
        self.custom_metrics = None
        self.custom_stopping_func = None

        self.train_sample_generation = data_bert.obtain_train_sample
        self.train_sample_square_generation = data_bert.obtain_train_sample

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False, actual_y=None, final_tree_level = None):
        if type(data) == dict:
            contents_description = data["contents"]["description"]
            contents_title = data["contents"]["title"]
            contents_lang = data["contents"]["lang"]
            topics_description = data["topics"]["description"]
            topics_title = data["topics"]["title"]
            topics_lang = data["topics"]["lang"]
            # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))

            contents_text_info = tf.concat([contents_description, contents_title], axis=-1)
            topics_text_info = tf.concat([topics_description, topics_title], axis=-1)

            shape = contents_description.shape
            is_ragged = len([1 for k in range(int(shape.rank)) if shape[k] is None]) > 0
            if is_ragged:
                first_layer1_contents = tf.ragged.map_flat_values(self.dropout0_contents, contents_text_info, training=training)
                first_layer1_topics = tf.ragged.map_flat_values(self.dropout0_topics, topics_text_info, training=training)
                first_layer2_contents = tf.ragged.map_flat_values(self.dropout0_feed_contents, contents_text_info, training=training)
                first_layer2_topics = tf.ragged.map_flat_values(self.dropout0_feed_topics, topics_text_info, training=training)
            else:
                first_layer1_contents = self.dropout0_contents(contents_text_info, training=training)
                first_layer1_topics = self.dropout0_topics(topics_text_info, training=training)
                first_layer2_contents = self.dropout0_feed_contents(contents_text_info, training=training)
                first_layer2_topics = self.dropout0_feed_topics(topics_text_info, training=training)

            first_layer1 = tf.concat([first_layer1_contents, contents_lang, first_layer1_topics, topics_lang], axis=-1)
            first_layer2 = tf.concat([first_layer2_contents, contents_lang, first_layer2_topics, topics_lang], axis=-1)
            embedding_result = self.concat_layer(
                [contents_description, contents_title, contents_lang, topics_description, topics_title, topics_lang])
        else:
            first_layer1 = data
            first_layer2 = data

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

        shape = first_layer1.shape
        if (training and actual_y is not None and actual_y.shape[0] is not None
            and actual_y.shape[0] == shape[0] and final_tree_level is not None and
                final_tree_level.shape[0] is not None and final_tree_level.shape[0] == shape[0]):
            p = 4.0
            tf_actual_y = tf.constant(actual_y, dtype=tf.float32)
            tf_not_actual_y = tf.constant(1 - actual_y, dtype=tf.float32)
            tf_final_tree_level = tf.expand_dims(tf.constant(final_tree_level, dtype=tf.float32), axis=1)
            tf_not_final_tree_level = tf.expand_dims(tf.constant(1 - final_tree_level, dtype=tf.float32), axis=1)

            tf_final_tree_level = tf.repeat(tf_final_tree_level, repeats=2, axis=1)
            tf_not_final_tree_level = tf.repeat(tf_not_final_tree_level, repeats=2, axis=1)

            t_max_probas = tf.reduce_max(t, axis=1)
            overshoot_max_probas = tf.reduce_max(overshoot_result, axis=1)
            """overshoot_max_probas_cpexpand = 100.0 * (tf.clip_by_value(tf.squeeze(overshoot_max_probas, axis=1),
                                                        clip_value_min=0.595, clip_value_max=0.605) - 0.595)"""
            
            tclip = tf.clip_by_value(t, clip_value_min=0.05, clip_value_max=0.95)
            pmean = tf.math.pow(tclip, p)
            pmean = tf.reduce_mean(pmean, axis=1)
            pmean = tf.squeeze(tf.math.pow(pmean, 1 / p), axis=1)

            pinvmean = tf.math.pow(tclip, p)
            pinvmean = tf.reduce_mean(pinvmean, axis=1)
            pinvmean = tf.clip_by_value(tf.squeeze(tf.math.pow(pinvmean, 1 / p), axis = 1),
                                        clip_value_min=0.05, clip_value_max=0.55)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.add(pinvmean * tf_actual_y, pmean * tf_not_actual_y) #* overshoot_max_probas_cpexpand

            osr = tf.clip_by_value(overshoot_result, clip_value_min=0.05, clip_value_max=0.95)
            pmean2 = tf.math.pow(osr, p)
            pmean2 = tf.reduce_mean(pmean2, axis=1)
            pmean2 = tf.squeeze(tf.math.pow(pmean2, 1 / p), axis=1)

            pinvmean2 = tf.math.pow(osr, p)
            pinvmean2 = tf.reduce_mean(pinvmean2, axis=1)
            pinvmean2 = tf.clip_by_value(tf.squeeze(tf.math.pow(pinvmean2, 1 / p), axis = 1),
                                        clip_value_min=0.05, clip_value_max=0.55)

            proba2 = tf.math.add(pinvmean2 * tf_actual_y, pmean2 * tf_not_actual_y)



            proba = tf.concat([tf.expand_dims(proba, axis = 1), tf.expand_dims(proba2, axis = 1)], axis=1)
            proba_simple = tf.concat([t_max_probas, overshoot_max_probas], axis = 1)
            return tf.math.add(proba * tf_not_final_tree_level, proba_simple * tf_final_tree_level)
        else:  # here we just return the probabilities normally. the probability will be computed as the max inside the set
            return tf.concat([tf.reduce_max(t, axis=1), tf.reduce_max(overshoot_result, axis=1)], axis = 1)
    def compile(self, weight_decay = 0.01, learning_rate = 0.0005):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000
        self.prev_entropy = None

        # saves the functions that generates whether a tuple (topic, content) is correlated.
        self.tuple_choice_sampler = data_bert_sampler.default_sampler_instance

        # generates the overshoot functions
        self.tuple_choice_sampler_overshoot = data_bert_sampler.default_sampler_overshoot2_instance

    # custom_generation_functions should be a dict, e.g. {"train_sample": data_bert.obtain_train_sample,
    #             "test_sample": data_bert.obtain_test_sample,
    #             "train_square_sample": data_bert.obtain_train_square_sample,
    #             "test_square_sample": data_bert.obtain_test_square_sample}
    def set_training_params(self, training_sample_size=15000, training_max_size=None,
                            training_sampler=None, custom_metrics=None, custom_stopping_func=None,
                            custom_tuple_choice_sampler=None, custom_tuple_choice_sampler_overshoot=None):
        self.training_sample_size = training_sample_size
        self.training_max_size = training_max_size
        if training_sampler is not None:
            self.training_sampler = training_sampler
        if custom_metrics is not None:
            custom_metrics.set_training_sampler(self.training_sampler)
            self.custom_metrics = custom_metrics
        if custom_stopping_func is not None:
            self.custom_stopping_func = custom_stopping_func
        if custom_tuple_choice_sampler is not None:
            self.tuple_choice_sampler = custom_tuple_choice_sampler
        if custom_tuple_choice_sampler_overshoot is not None:
            self.tuple_choice_sampler_overshoot = custom_tuple_choice_sampler_overshoot

    def train_step_tree(self):
        for k in range(50):
            # two pass, we first compute on overshoot only, and then compute on the full thing
            ratio1 = 2000.0 / 4000
            ratio2 = 2000.0 / 4000

            topics, contents, cors, class_ids, tree_levels, multipliers = self.tuple_choice_sampler.obtain_train_sample(
                int(ratio1 * self.training_sample_size))
            topics2, contents2, cors2, class_ids2, tree_levels2, multipliers2 = self.tuple_choice_sampler_overshoot.obtain_train_notree_sample(
                int(ratio2 * self.training_sample_size))

            y0 = np.tile(np.concatenate([cors, cors2]), 2)

            topics = np.concatenate([topics, topics2])
            contents = np.concatenate([contents, contents2])
            tree_levels = np.concatenate([tree_levels, tree_levels2])

            input_data = self.training_sampler.obtain_input_data_tree_both(topics_id_klevel=topics, contents_id=contents,
                                                                           tree_levels=tree_levels, final_level=5)
            y = tf.constant(y0)
            final_tree_level = np.tile((tree_levels == 5).astype(dtype=np.float32), 2)

            multipliers_tf = tf.constant(np.tile(
                np.concatenate([multipliers, multipliers2 * 10], axis=0),
                2), dtype=tf.float32)

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True, actual_y = y0, final_tree_level = final_tree_level)
                y_pred = tf.concat([y_pred[:len(cors), 0], y_pred[len(cors):len(y0), 1],
                           y_pred[len(y0):(len(y0) + len(cors)), 0], y_pred[(len(y0) + len(cors)):(2 * len(y0)), 1]],
                          axis=0)
                loss = self.loss(tf.expand_dims(y, axis=1), tf.expand_dims(y_pred, axis=1), sample_weight=multipliers_tf)
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.training_max_size is None:
            limit = 9223372036854775807
            limit_sq = 9223372036854775807
        else:
            limit = self.training_max_size

        # evaluation at larger subset
        ratio1 = 2000.0 / 4000
        ratio2 = 2000.0 / 4000

        topics, contents, cors, class_ids, tree_levels, multipliers = self.tuple_choice_sampler.obtain_train_sample(
            int(ratio1 * self.training_sample_size))
        topics2, contents2, cors2, class_ids2, tree_levels2, multipliers2 = self.tuple_choice_sampler_overshoot.obtain_train_notree_sample(
            int(ratio2 * self.training_sample_size))

        y0 = np.tile(np.concatenate([cors, cors2]), 2)

        topics = np.concatenate([topics, topics2])
        contents = np.concatenate([contents, contents2])
        tree_levels = np.concatenate([tree_levels, tree_levels2])

        input_data = self.training_sampler.obtain_input_data_tree_both(topics_id_klevel=topics, contents_id=contents,
                                                                       tree_levels=tree_levels, final_level=5)
        y = tf.constant(y0)
        final_tree_level = np.tile((tree_levels == 5).astype(dtype=np.float32), 2)

        multipliers_tf = tf.constant(np.tile(
            np.concatenate([multipliers, multipliers2 * 10], axis=0),
            2), dtype=tf.float32)

        y_pred = self(input_data, training=True, actual_y=y0, final_tree_level=final_tree_level)
        y_pred = tf.concat([y_pred[:len(cors), 0], y_pred[len(cors):len(y0), 1],
                            y_pred[len(y0):(len(y0)+len(cors)), 0], y_pred[(len(y0)+len(cors)):(2*len(y0)), 1]], axis = 0)
        self.entropy_large_set.update_state(tf.expand_dims(y, axis=1), tf.expand_dims(y_pred, axis=1), sample_weight=multipliers_tf)

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

    def train_step(self, data):
        if self.tuple_choice_sampler.is_tree_sampler():
            return self.train_step_tree()
        for k in range(50):
            # two pass, we first compute on overshoot only, and then compute on the full thing
            ratio1 = 500.0 / 4000
            ratio2 = 3500.0 / 4000

            topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(int(ratio1 * self.training_sample_size))
            topics2, contents2, cors, class_ids2 = self.tuple_choice_sampler_overshoot.obtain_train_sample(int(ratio2 * self.training_sample_size))

            y0_1 = self.tuple_choice_sampler.has_correlations(contents, topics, class_ids)

            y1_2 = self.tuple_choice_sampler_overshoot.has_correlations(contents2, topics2, class_ids2)

            y0 = np.tile(np.concatenate([y0_1, y1_2]), 2)

            topics = np.concatenate([topics, topics2])
            contents = np.concatenate([contents, contents2])

            input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)

            y = tf.constant(y0)

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True)
                loss = self.loss(y, tf.concat([y_pred[:len(y0_1), 0], y_pred[len(y0_1):len(y0), 1],
                            y_pred[len(y0):(len(y0)+len(y0_1)), 0], y_pred[(len(y0)+len(y0_1)):(2*len(y0)), 1]], axis = 0))
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.training_max_size is None:
            limit = 9223372036854775807
            limit_sq = 9223372036854775807
        else:
            limit = self.training_max_size

        # evaluation at larger subset
        ratio1 = 500.0 / 4000
        ratio2 = 3500.0 / 4000

        topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(
            int(ratio1 * self.training_sample_size))
        topics2, contents2, cors, class_ids2 = self.tuple_choice_sampler_overshoot.obtain_train_sample(
            int(ratio2 * self.training_sample_size))

        y0_1 = self.tuple_choice_sampler.has_correlations(contents, topics, class_ids)

        y1_2 = self.tuple_choice_sampler_overshoot.has_correlations(contents2, topics2, class_ids2)

        y0 = np.tile(np.concatenate([y0_1, y1_2]), 2)

        topics = np.concatenate([topics, topics2])
        contents = np.concatenate([contents, contents2])

        input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)

        y = tf.constant(y0)

        y_pred = self(input_data)
        y_pred = tf.concat([y_pred[:len(y0_1), 0], y_pred[len(y0_1):len(y0), 1],
                            y_pred[len(y0):(len(y0)+len(y0_1)), 0], y_pred[(len(y0)+len(y0_1)):(2*len(y0)), 1]], axis = 0)
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

    def __init__(self):
        model_bert_fix.CustomMetrics.__init__(self)
        self.metrics = [] # a lists of dicts, containing the metrics, and the data_bert_sampler.SamplerBase which contains the metric
        self.tree_metrics = []

    def add_metric(self, name, tuple_choice_sampler, tuple_choice_sampler_overshoot, sample_choice = TEST, threshold = 0.5):
        accuracy = tf.keras.metrics.BinaryAccuracy(name = name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name = name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name = name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name = name + "_entropy")

        accuracy_nolang = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy_nolang", threshold=threshold)
        precision_nolang = tf.keras.metrics.Precision(name=name + "_precision_nolang", thresholds=threshold)
        recall_nolang = tf.keras.metrics.Recall(name=name + "_recall_nolang", thresholds=threshold)
        entropy_nolang = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy_nolang")
        self.metrics.append({"metrics": [accuracy, precision, recall, entropy, accuracy_nolang, precision_nolang, recall_nolang, entropy_nolang], "sampler": tuple_choice_sampler, "sampler_overshoot": tuple_choice_sampler_overshoot, "sample_choice": sample_choice})

    def add_tree_metric(self, name, tuple_choice_sampler_tree, level, sample_choice = TEST, threshold = 0.7):
        accuracy = tf.keras.metrics.BinaryAccuracy(name = name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name = name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name = name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name = name + "_entropy")

        accuracy_nolang = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy_nolang", threshold=threshold)
        precision_nolang = tf.keras.metrics.Precision(name=name + "_precision_nolang", thresholds=threshold)
        recall_nolang = tf.keras.metrics.Recall(name=name + "_recall_nolang", thresholds=threshold)
        entropy_nolang = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy_nolang")
        self.tree_metrics.append({"metrics": [accuracy, precision, recall, entropy, accuracy_nolang, precision_nolang, recall_nolang, entropy_nolang], "sampler": tuple_choice_sampler_tree, "sample_choice": sample_choice, "level": level})

    def update_metrics(self, model, sample_size_limit):
        for k in range(len(self.metrics)):
            kmetrics = self.metrics[k]["metrics"]
            sampler = self.metrics[k]["sampler"]
            sampler_overshoot = self.metrics[k]["sampler_overshoot"]
            sample_choice = self.metrics[k]["sample_choice"]

            if sample_choice <= 4:
                if sample_choice == DynamicMetrics.TRAIN:
                    topics, contents, cors, class_id = sampler.obtain_train_sample(min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TRAIN_SQUARE:
                    topics, contents, cors, class_id = sampler.obtain_train_square_sample(min(360000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST:
                    topics, contents, cors, class_id = sampler.obtain_test_sample(min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_SQUARE:
                    topics, contents, cors, class_id = sampler.obtain_test_square_sample(min(360000, sample_size_limit))
                input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
                y = tf.constant(cors, dtype = tf.float32)
                y_pred = model(input_data)[:, 0]
            else:
                if sample_choice == DynamicMetrics.TRAIN_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_train_sample(min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TRAIN_SQUARE_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_train_square_sample(min(360000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_test_sample(min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_SQUARE_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_test_square_sample(min(360000, sample_size_limit))
                input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
                y = tf.constant(cors, dtype = tf.float32)
                y_pred = model(input_data)[:, 1]

            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
            if sample_choice <= 4:
                y_pred = model(input_data)[:, 0]
            else:
                y_pred = model(input_data)[:, 1]
            for j in range(4,8):
                kmetrics[j].update_state(y, y_pred)
            gc.collect()

        for k in range(len(self.tree_metrics)):
            kmetrics = self.tree_metrics[k]["metrics"]
            sampler = self.tree_metrics[k]["sampler"]
            sample_choice = self.tree_metrics[k]["sample_choice"]
            level = self.tree_metrics[k]["level"]

            if sample_choice <= 4:
                if sample_choice == DynamicMetrics.TRAIN:
                    topics, contents, cors = sampler.obtain_tree_train_sample(min(50, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TRAIN_SQUARE:
                    topics, contents, cors = sampler.obtain_tree_train_square_sample(min(7, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TEST:
                    topics, contents, cors = sampler.obtain_tree_test_sample(min(50, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TEST_SQUARE:
                    topics, contents, cors = sampler.obtain_tree_test_square_sample(min(7, sample_size_limit), level)
            else:
                if sample_choice == DynamicMetrics.TRAIN_OVERSHOOT:
                    topics, contents, cors = sampler.obtain_tree_train_sample(min(50, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TRAIN_SQUARE_OVERSHOOT:
                    topics, contents, cors = sampler.obtain_tree_train_square_sample(min(7, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TEST_OVERSHOOT:
                    topics, contents, cors = sampler.obtain_tree_test_sample(min(50, sample_size_limit), level)
                elif sample_choice == DynamicMetrics.TEST_SQUARE_OVERSHOOT:
                    topics, contents, cors = sampler.obtain_tree_test_square_sample(min(7, sample_size_limit), level)
            input_data = self.training_sampler.obtain_input_data_tree(topics_id_klevel=topics, contents_id=contents,
                                                                      tree_levels=np.repeat(level, len(cors)),final_level=5)
            y = tf.constant(cors, dtype = tf.float32)
            if sample_choice <= 4:
                y_pred = model(input_data)[:,0]
            else:
                y_pred = model(input_data)[:,1]
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_tree_filter_lang(topics_id_klevel=topics, contents_id=contents,
                                                                      tree_levels=np.repeat(level, len(cors)),final_level=5)
            if sample_choice <= 4:
                y_pred = model(input_data)[:, 0]
            else:
                y_pred = model(input_data)[:, 1]
            for j in range(4,8):
                kmetrics[j].update_state(y, y_pred)
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

    def get_total_tree_metric(self):
        sum = 0.0
        for k in range(len(self.tree_metrics)):
            mmetrics = self.tree_metrics[k]["metrics"]
            sum += mmetrics[3].result()
        return sum + self.get_test_entropy_metric()

default_metrics = DynamicMetrics()
default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST)
default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE)
default_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_metrics.add_metric("test_square_overshoot", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE_OVERSHOOT)

default_tree_metrics = DynamicMetrics()
default_tree_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE)
default_tree_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv0", data_bert_sampler.default_tree_sampler_instance, level = 0, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv1", data_bert_sampler.default_tree_sampler_instance, level = 1, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv2", data_bert_sampler.default_tree_sampler_instance, level = 2, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv3", data_bert_sampler.default_tree_sampler_instance, level = 3, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv4", data_bert_sampler.default_tree_sampler_instance, level = 4, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv0_stringent", data_bert_sampler.default_tree_sampler_instance, level = 0, sample_choice = DynamicMetrics.TEST, threshold=0.8)
default_tree_metrics.add_tree_metric("treelv1_stringent", data_bert_sampler.default_tree_sampler_instance, level = 1, sample_choice = DynamicMetrics.TEST, threshold=0.8)
default_tree_metrics.add_tree_metric("treelv2_stringent", data_bert_sampler.default_tree_sampler_instance, level = 2, sample_choice = DynamicMetrics.TEST, threshold=0.8)
default_tree_metrics.add_tree_metric("treelv3_stringent", data_bert_sampler.default_tree_sampler_instance, level = 3, sample_choice = DynamicMetrics.TEST, threshold=0.8)
default_tree_metrics.add_tree_metric("treelv4_stringent", data_bert_sampler.default_tree_sampler_instance, level = 4, sample_choice = DynamicMetrics.TEST, threshold=0.8)

"""
default_tree_metrics.add_tree_metric("treelv0_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 0, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv1_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 1, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv2_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 2, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv3_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 3, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv4_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 4, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
"""

def obtain_overshoot_metric_instance(training_tuple_sampler, training_tuple_sampler_overshoot):
    overshoot_metrics = DynamicMetrics()
    overshoot_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice=DynamicMetrics.TEST_SQUARE)
    overshoot_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance,
                               data_bert_sampler.default_sampler_overshoot2_instance,
                               sample_choice=DynamicMetrics.TEST_OVERSHOOT)
    overshoot_metrics.add_metric("test_square_overshoot", data_bert_sampler.default_sampler_instance,
                               data_bert_sampler.default_sampler_overshoot2_instance,
                               sample_choice=DynamicMetrics.TEST_SQUARE_OVERSHOOT)
    overshoot_metrics.add_metric("test_in_train_sample", training_tuple_sampler, training_tuple_sampler_overshoot, sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square_in_train_sample", training_tuple_sampler, training_tuple_sampler_overshoot, sample_choice=DynamicMetrics.TEST_SQUARE)
    return overshoot_metrics

class DefaultStoppingFunc(model_bert_fix.CustomStoppingFunc):
    def __init__(self, model_dir):
        model_bert_fix.CustomStoppingFunc.__init__(self, model_dir)
        self.lowest_test_small_entropy = None
        self.countdown = 0
        self.equal_thresh = 0

    def evaluate(self, custom_metrics, model):
        if self.lowest_test_small_entropy is not None:
            current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
                self.equal_thresh = 0
            elif current_test_small_entropy > self.lowest_test_small_entropy * 1.02:
                self.countdown += 1
                if self.countdown > 10:
                    return False

        else:
            current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            self.lowest_test_small_entropy = current_test_small_entropy

    def force_final_state(self):
        self.lowest_test_small_entropy = None
        self.countdown = 0
        self.equal_thresh = 0

class DefaultTreeStoppingFunc(model_bert_fix.CustomStoppingFunc):
    def __init__(self, model_dir):
        model_bert_fix.CustomStoppingFunc.__init__(self, model_dir)
        self.lowest_test_small_entropy = None
        self.countdown = 0

    def evaluate(self, custom_metrics, model):
        if self.lowest_test_small_entropy is not None:
            current_test_small_entropy = custom_metrics.get_total_tree_metric()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
            elif current_test_small_entropy > self.lowest_test_small_entropy * 1.3:
                self.countdown += 1
                if self.countdown > 50:
                    return True
        else:
            current_test_small_entropy = custom_metrics.get_total_tree_metric()
            self.lowest_test_small_entropy = current_test_small_entropy
        return False