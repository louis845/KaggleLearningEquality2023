import data_bert
import data_bert_tree_struct
import model_bert_fix
import tensorflow as tf
import numpy as np
import config
import gc
import data_bert_sampler
import model_bert_submodels


#

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise_topics=0.05, init_noise_overshoot_topics=0.2,
                 init_noise_contents=0.05, init_noise_overshoot_contents=0.2,
                 init_noise_lang=0.02, init_noise_overshoot_lang=0.3):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # standard stuff
        self.dropout0_topics = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_topics)
        self.dropout0_feed_topics = tf.keras.layers.GaussianNoise(stddev=init_noise_topics)
        self.dropout0_contents = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_contents)
        self.dropout0_feed_contents = tf.keras.layers.GaussianNoise(stddev=init_noise_contents)
        self.dropout0_lang = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot_lang)
        self.dropout0_feed_lang = tf.keras.layers.GaussianNoise(stddev=init_noise_lang)

        self.dim_reduce_model_overshoot = model_bert_submodels.SiameseTwinSmallSubmodel(units_size=units_size, name="dim_red_os",
                                                                           left_name="dim_red_os_left",
                                                                           right_name="dim_red_os_right",
                                                                           final_layer_size=256)

        self.dim_reduce_model_final = model_bert_submodels.SiameseTwinSmallSubmodel(units_size=units_size, name="dim_red_final",
                                                                         left_name="dim_red_os_left",
                                                                         right_name="dim_red_os_right",
                                                                         final_layer_size=256)

        self.stepup_submodel = model_bert_submodels.StepupSubmodel(units_size=units_size)

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
    def call(self, data, training=False, actual_y=None, final_tree_level=None):
        # APPLY DIFFERENT NOISE to the two input layers.
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
            first_layer1_contents = tf.ragged.map_flat_values(self.dropout0_contents, contents_text_info,
                                                              training=training)
            first_layer1_topics = tf.ragged.map_flat_values(self.dropout0_topics, topics_text_info,
                                                            training=training)
            first_layer1_contents_lang = tf.ragged.map_flat_values(self.dropout0_lang, contents_lang,
                                                                   training=training)
            first_layer1_topics_lang = tf.ragged.map_flat_values(self.dropout0_lang, topics_lang, training=training)

            first_layer2_contents = tf.ragged.map_flat_values(self.dropout0_feed_contents, contents_text_info,
                                                              training=training)
            first_layer2_topics = tf.ragged.map_flat_values(self.dropout0_feed_topics, topics_text_info,
                                                            training=training)
            first_layer2_contents_lang = tf.ragged.map_flat_values(self.dropout0_feed_lang, contents_lang,
                                                                   training=training)
            first_layer2_topics_lang = tf.ragged.map_flat_values(self.dropout0_feed_lang, topics_lang,
                                                                 training=training)
        else:
            first_layer1_contents = self.dropout0_contents(contents_text_info, training=training)
            first_layer1_topics = self.dropout0_topics(topics_text_info, training=training)
            first_layer1_contents_lang = self.dropout0_lang(contents_lang, training=training)
            first_layer1_topics_lang = self.dropout0_lang(topics_lang, training=training)

            first_layer2_contents = self.dropout0_feed_contents(contents_text_info, training=training)
            first_layer2_topics = self.dropout0_feed_topics(topics_text_info, training=training)
            first_layer2_contents_lang = self.dropout0_feed_lang(contents_lang, training=training)
            first_layer2_topics_lang = self.dropout0_feed_lang(topics_lang, training=training)

        # original scheme is [contents, contents_lang, topics, topics_lang]
        first_layer1_contents = tf.concat(
            [first_layer1_contents, first_layer1_contents_lang],
            axis=-1)
        first_layer1_topics = tf.concat(
            [first_layer1_topics, first_layer1_topics_lang],
            axis=-1)
        first_layer2_contents = tf.concat(
            [first_layer2_contents, first_layer2_contents_lang],
            axis=-1)
        first_layer2_topics = tf.concat(
            [first_layer2_topics, first_layer2_topics_lang],
            axis=-1)

        first_layer1 = {"left": first_layer1_contents,
                        "right": first_layer1_topics}
        first_layer2 = {"left": first_layer2_contents,
                        "right": first_layer2_topics}

        ovs_result = self.stepup_submodel(
            {"first_layer1":self.dim_reduce_model_overshoot(first_layer1),
             "first_layer2":self.dim_reduce_model_final(first_layer2)}
            ,training=training)
        t = ovs_result["t"]
        overshoot_result = ovs_result["overshoot_result"]

        shape = first_layer1_contents.shape
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
            pinvmean = tf.clip_by_value(tf.squeeze(tf.math.pow(pinvmean, 1 / p), axis=1),
                                        clip_value_min=0.05, clip_value_max=0.55)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.add(pinvmean * tf_actual_y, pmean * tf_not_actual_y)  # * overshoot_max_probas_cpexpand

            osr = tf.clip_by_value(overshoot_result, clip_value_min=0.05, clip_value_max=0.95)
            pmean2 = tf.math.pow(osr, p)
            pmean2 = tf.reduce_mean(pmean2, axis=1)
            pmean2 = tf.squeeze(tf.math.pow(pmean2, 1 / p), axis=1)

            pinvmean2 = tf.math.pow(osr, p)
            pinvmean2 = tf.reduce_mean(pinvmean2, axis=1)
            pinvmean2 = tf.clip_by_value(tf.squeeze(tf.math.pow(pinvmean2, 1 / p), axis=1),
                                         clip_value_min=0.05, clip_value_max=0.55)

            proba2 = tf.math.add(pinvmean2 * tf_actual_y, pmean2 * tf_not_actual_y)

            proba = tf.concat([tf.expand_dims(proba, axis=1), tf.expand_dims(proba2, axis=1)], axis=1)
            proba_simple = tf.concat([t_max_probas, overshoot_max_probas], axis=1)
            return tf.math.add(proba * tf_not_final_tree_level, proba_simple * tf_final_tree_level)
        else:  # here we just return the probabilities normally. the probability will be computed as the max inside the set
            return tf.concat([tf.reduce_max(t, axis=1), tf.reduce_max(overshoot_result, axis=1)], axis=1)

    def compile(self, weight_decay=0.01, learning_rate=0.0005):
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

            input_data = self.training_sampler.obtain_input_data_tree_both(topics_id_klevel=topics,
                                                                           contents_id=contents,
                                                                           tree_levels=tree_levels, final_level=5)
            y = tf.constant(y0)
            final_tree_level = np.tile((tree_levels == 5).astype(dtype=np.float32), 2)

            multipliers_tf = tf.constant(np.tile(
                np.concatenate([multipliers, multipliers2 * 10], axis=0),
                2), dtype=tf.float32)

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True, actual_y=y0, final_tree_level=final_tree_level)
                y_pred = tf.concat([y_pred[:len(cors), 0], y_pred[len(cors):len(y0), 1],
                                    y_pred[len(y0):(len(y0) + len(cors)), 0],
                                    y_pred[(len(y0) + len(cors)):(2 * len(y0)), 1]],
                                   axis=0)
                loss = self.loss(tf.expand_dims(y, axis=1), tf.expand_dims(y_pred, axis=1),
                                 sample_weight=multipliers_tf)
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
                            y_pred[len(y0):(len(y0) + len(cors)), 0], y_pred[(len(y0) + len(cors)):(2 * len(y0)), 1]],
                           axis=0)
        self.entropy_large_set.update_state(tf.expand_dims(y, axis=1), tf.expand_dims(y_pred, axis=1),
                                            sample_weight=multipliers_tf)

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

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True)
                loss = self.loss(y, tf.concat([y_pred[:len(y0_1), 0], y_pred[len(y0_1):len(y0), 1],
                                               y_pred[len(y0):(len(y0) + len(y0_1)), 0],
                                               y_pred[(len(y0) + len(y0_1)):(2 * len(y0)), 1]], axis=0))
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
                            y_pred[len(y0):(len(y0) + len(y0_1)), 0], y_pred[(len(y0) + len(y0_1)):(2 * len(y0)), 1]],
                           axis=0)
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