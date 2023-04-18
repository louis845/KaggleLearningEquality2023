import data_bert
import data_bert_tree_struct
import model_bert_fix
import tensorflow as tf
import numpy as np
import config
import gc
import data_bert_sampler
import model_bert_submodels


# same as stepup class, but we have 3 stepup layers, and use a modular approach. tree training is not supported.

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512, init_noise=0.05, init_noise_overshoot=0.2, init_noise_overshoot2=0.6, use_siamese = False):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        self.use_siamese = use_siamese
        if use_siamese:
            self.overshoot2_layer = model_bert_submodels.SiameseTwinSubmodel(units_size=units_size)
            self.overshoot_layer = model_bert_submodels.SiameseTwinSubmodel(units_size=units_size)
            self.usual_layer = model_bert_submodels.SiameseTwinSubmodel(units_size=units_size)

            self.dropout0_overshoot2_left = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot2)
            self.dropout0_overshoot_left = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot)
            self.dropout0_left = tf.keras.layers.GaussianNoise(stddev=init_noise)

            self.dropout0_overshoot2_right = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot2)
            self.dropout0_overshoot_right = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot)
            self.dropout0_right = tf.keras.layers.GaussianNoise(stddev=init_noise)
        else:
            self.overshoot2_layer = model_bert_submodels.FullyConnectedSubmodel(units_size=units_size)
            self.overshoot_layer = model_bert_submodels.FullyConnectedSubmodel(units_size=units_size)
            self.usual_layer = model_bert_submodels.FullyConnectedSubmodel(units_size=units_size)

            self.dropout0_overshoot2 = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot2)
            self.dropout0_overshoot = tf.keras.layers.GaussianNoise(stddev=init_noise_overshoot)
            self.dropout0 = tf.keras.layers.GaussianNoise(stddev=init_noise)

            self.concat_layer = tf.keras.layers.Concatenate(axis=-1)

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
    def call(self, data, training=False, actual_y=None, final_tree_level=None):
        if self.use_siamese:
            os2result = self.overshoot2_layer(data)
            os2result[""]
        else:


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
        self.metrics = []  # a lists of dicts, containing the metrics, and the data_bert_sampler.SamplerBase which contains the metric
        self.tree_metrics = []

    def add_metric(self, name, tuple_choice_sampler, tuple_choice_sampler_overshoot, sample_choice=TEST, threshold=0.5):
        accuracy = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name=name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name=name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy")

        accuracy_nolang = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy_nolang", threshold=threshold)
        precision_nolang = tf.keras.metrics.Precision(name=name + "_precision_nolang", thresholds=threshold)
        recall_nolang = tf.keras.metrics.Recall(name=name + "_recall_nolang", thresholds=threshold)
        entropy_nolang = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy_nolang")
        self.metrics.append({"metrics": [accuracy, precision, recall, entropy, accuracy_nolang, precision_nolang,
                                         recall_nolang, entropy_nolang], "sampler": tuple_choice_sampler,
                             "sampler_overshoot": tuple_choice_sampler_overshoot, "sample_choice": sample_choice})

    def add_tree_metric(self, name, tuple_choice_sampler_tree, level, sample_choice=TEST, threshold=0.6):
        accuracy = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name=name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name=name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy")

        accuracy_nolang = tf.keras.metrics.BinaryAccuracy(name=name + "_accuracy_nolang", threshold=threshold)
        precision_nolang = tf.keras.metrics.Precision(name=name + "_precision_nolang", thresholds=threshold)
        recall_nolang = tf.keras.metrics.Recall(name=name + "_recall_nolang", thresholds=threshold)
        entropy_nolang = tf.keras.metrics.BinaryCrossentropy(name=name + "_entropy_nolang")
        self.tree_metrics.append({"metrics": [accuracy, precision, recall, entropy, accuracy_nolang, precision_nolang,
                                              recall_nolang, entropy_nolang], "sampler": tuple_choice_sampler_tree,
                                  "sample_choice": sample_choice, "level": level})

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
                    topics, contents, cors, class_id = sampler.obtain_train_square_sample(
                        min(360000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST:
                    topics, contents, cors, class_id = sampler.obtain_test_sample(min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_SQUARE:
                    topics, contents, cors, class_id = sampler.obtain_test_square_sample(min(360000, sample_size_limit))
                input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
                y = tf.constant(cors, dtype=tf.float32)
                y_pred = model(input_data)[:, 0]
            else:
                if sample_choice == DynamicMetrics.TRAIN_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_train_sample(
                        min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TRAIN_SQUARE_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_train_square_sample(
                        min(360000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_test_sample(
                        min(60000, sample_size_limit))
                elif sample_choice == DynamicMetrics.TEST_SQUARE_OVERSHOOT:
                    topics, contents, cors, class_id = sampler_overshoot.obtain_test_square_sample(
                        min(360000, sample_size_limit))
                input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
                y = tf.constant(cors, dtype=tf.float32)
                y_pred = model(input_data)[:, 1]

            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
            if sample_choice <= 4:
                y_pred = model(input_data)[:, 0]
            else:
                y_pred = model(input_data)[:, 1]
            for j in range(4, 8):
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
                                                                      tree_levels=np.repeat(level, len(cors)),
                                                                      final_level=5)
            y = tf.constant(cors, dtype=tf.float32)
            if sample_choice <= 4:
                y_pred = model(input_data)[:, 0]
            else:
                y_pred = model(input_data)[:, 1]
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_tree_filter_lang(topics_id_klevel=topics,
                                                                                  contents_id=contents,
                                                                                  tree_levels=np.repeat(level,
                                                                                                        len(cors)),
                                                                                  final_level=5)
            if sample_choice <= 4:
                y_pred = model(input_data)[:, 0]
            else:
                y_pred = model(input_data)[:, 1]
            for j in range(4, 8):
                kmetrics[j].update_state(y, y_pred)

    def obtain_metrics(self):
        metrics_list = [metr for met in self.metrics for metr in met["metrics"]] + [metr for met in self.tree_metrics
                                                                                    for metr in met["metrics"]]
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
default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance,
                           data_bert_sampler.default_sampler_overshoot2_instance, sample_choice=DynamicMetrics.TEST)
default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance,
                           data_bert_sampler.default_sampler_overshoot2_instance,
                           sample_choice=DynamicMetrics.TEST_SQUARE)
default_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance,
                           data_bert_sampler.default_sampler_overshoot2_instance,
                           sample_choice=DynamicMetrics.TEST_OVERSHOOT)
default_metrics.add_metric("test_square_overshoot", data_bert_sampler.default_sampler_instance,
                           data_bert_sampler.default_sampler_overshoot2_instance,
                           sample_choice=DynamicMetrics.TEST_SQUARE_OVERSHOOT)

default_tree_metrics = DynamicMetrics()
default_tree_metrics.add_metric("test", data_bert_sampler.default_sampler_instance,
                                data_bert_sampler.default_sampler_overshoot2_instance,
                                sample_choice=DynamicMetrics.TEST)
default_tree_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance,
                                data_bert_sampler.default_sampler_overshoot2_instance,
                                sample_choice=DynamicMetrics.TEST_SQUARE)
default_tree_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance,
                                data_bert_sampler.default_sampler_overshoot2_instance,
                                sample_choice=DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv0", data_bert_sampler.default_tree_sampler_instance, level=0,
                                     sample_choice=DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv1", data_bert_sampler.default_tree_sampler_instance, level=1,
                                     sample_choice=DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv2", data_bert_sampler.default_tree_sampler_instance, level=2,
                                     sample_choice=DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv3", data_bert_sampler.default_tree_sampler_instance, level=3,
                                     sample_choice=DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv4", data_bert_sampler.default_tree_sampler_instance, level=4,
                                     sample_choice=DynamicMetrics.TEST)

"""
default_tree_metrics.add_tree_metric("treelv0_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 0, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv1_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 1, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv2_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 2, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv3_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 3, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_tree_metrics.add_tree_metric("treelv4_overshoot", data_bert_sampler.default_tree_sampler_instance, level = 4, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
"""


def obtain_overshoot_metric_instance(training_tuple_sampler, training_tuple_sampler_overshoot):
    overshoot_metrics = DynamicMetrics()
    overshoot_metrics.add_metric("test", data_bert_sampler.default_sampler_instance,
                                 data_bert_sampler.default_sampler_overshoot2_instance,
                                 sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance,
                                 data_bert_sampler.default_sampler_overshoot2_instance,
                                 sample_choice=DynamicMetrics.TEST_SQUARE)
    overshoot_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance,
                                 data_bert_sampler.default_sampler_overshoot2_instance,
                                 sample_choice=DynamicMetrics.TEST_OVERSHOOT)
    overshoot_metrics.add_metric("test_square_overshoot", data_bert_sampler.default_sampler_instance,
                                 data_bert_sampler.default_sampler_overshoot2_instance,
                                 sample_choice=DynamicMetrics.TEST_SQUARE_OVERSHOOT)
    overshoot_metrics.add_metric("test_in_train_sample", training_tuple_sampler, training_tuple_sampler_overshoot,
                                 sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square_in_train_sample", training_tuple_sampler,
                                 training_tuple_sampler_overshoot, sample_choice=DynamicMetrics.TEST_SQUARE)
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
                    return True

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