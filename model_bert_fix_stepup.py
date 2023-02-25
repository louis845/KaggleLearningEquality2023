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
    def __init__(self, units_size=512):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis=2, name = "initial_concat")

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianDropout(rate=0.2)

        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name="dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.1)
        # the results of dense2 will be plugged into this.
        self.denseOvershoot = tf.keras.layers.Dense(units=units_size // 4, activation="relu", name = "denseOvershoot")
        self.dropoutOvershoot = tf.keras.layers.Dropout(rate=0.1)
        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid", name = "finalOvershoot")

        # dense1_fp takes in the combined input of dense0 and denseOvershoot
        self.dense1_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense1_fp")
        self.dropout1_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense2_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense2_fp")
        self.dropout2_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense3_fp = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense3_fp")
        self.dropout3_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense4_fp = tf.keras.layers.Dense(units=128, name = "dense4_fp")
        self.relu4 = tf.keras.layers.ReLU()
        self.dropout4_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense5 = tf.keras.layers.Dense(units=1, activation="sigmoid", name = "dense5")

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

        self.state_is_final = False

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size x 2) numpy array, where
    # the first column is for the full model prediction array, the second column for the intermediate
    # prediction
    def call(self, data, training=False, actual_y=None):
        if type(data) == dict:
            contents_description = data["contents"]["description"]
            contents_title = data["contents"]["title"]
            contents_lang = data["contents"]["lang"]
            topics_description = data["topics"]["description"]
            topics_title = data["topics"]["title"]
            topics_lang = data["topics"]["lang"]
            # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))
            embedding_result = self.concat_layer(
                [contents_description, contents_title, contents_lang, topics_description, topics_title, topics_lang])
        else:
            embedding_result = data

        first_layer = self.dropout0(embedding_result, training=training)
        t = self.dropout1(self.dense1(first_layer), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        res_dropout4 = self.dropout4(self.dense4(t), training=training)

        overshoot_fullresult = self.dropoutOvershoot(self.denseOvershoot(res_dropout4), training=training)
        overshoot_result = self.finalOvershoot(overshoot_fullresult)


        t = self.dropout1_fp(self.dense1_fp(tf.concat([first_layer, overshoot_fullresult], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3_fp(self.dense3_fp(t), training=training)
        t = self.dropout4_fp(self.relu4(self.dense4_fp(t)), training=training)
        t = self.dense5(t)
        if training and actual_y is not None:  # here we use overestimation training method for the set
            p = 4.0
            pmean = tf.math.pow(t, p)
            pmean = tf.reduce_mean(pmean, axis=1)
            pmean = tf.squeeze(tf.math.pow(pmean, 1 / p), axis=1)

            pinvmean = tf.math.pow(t, 1 / p)
            pinvmean = tf.reduce_mean(pinvmean, axis=1)
            pinvmean = tf.math.pow(pinvmean, p)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.multiply(pinvmean * tf.constant(actual_y), pmean * tf.constant(1 - actual_y))

            pmean2 = tf.math.pow(overshoot_result, p)
            pmean2 = tf.reduce_mean(pmean2, axis=1)
            pmean2 = tf.squeeze(tf.math.pow(pmean2, 1 / p), axis=1)

            pinvmean2 = tf.math.pow(overshoot_result, 1 / p)
            pinvmean2 = tf.reduce_mean(pinvmean2, axis=1)
            pinvmean2 = tf.math.pow(pinvmean2, p)

            combined_pmean = tf.concat([pmean, pmean2], axis = 1)
            combined_pinvmean = tf.concat([pinvmean, pinvmean2], axis=1)

            proba = tf.math.add(pinvmean * tf.constant(actual_y), pmean2 * tf.constant(1 - actual_y))
            return proba
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

    def train_step(self, data):
        for k in range(50):
            # two pass, we first compute on overshoot only, and then compute on the full thing
            if not self.state_is_final:
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
            else:
                ratio1 = 1.0 / 4000
                ratio2 = 3999.0 / 4000

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
        if not self.state_is_final:
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

            y_pred = self(input_data, training=True)
            y_pred = tf.concat([y_pred[:len(y0_1), 0], y_pred[len(y0_1):len(y0), 1],
                                y_pred[len(y0):(len(y0)+len(y0_1)), 0], y_pred[(len(y0)+len(y0_1)):(2*len(y0)), 1]], axis = 0)
            self.entropy_large_set.update_state(y, y_pred)
        else:
            ratio1 = 1.0 / 4000
            ratio2 = 3999.0 / 4000

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

            y_pred = self(input_data, training=True)
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

        # Return a dict mapping metric names to current value
        temp_final_val = 0.7 if self.state_is_final else 0.3
        return {"is_final":temp_final_val, **{m.name: m.result() for m in self.metrics},
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
                y = tf.constant(cors)
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
                y = tf.constant(cors)
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
    def obtain_metrics(self):
        metrics_list = [metr for met in self.metrics for metr in met["metrics"]]
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

default_metrics = DynamicMetrics()
default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST)
default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE)
default_metrics.add_metric("test_overshoot", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_OVERSHOOT)
default_metrics.add_metric("test_square_overshoot", data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, sample_choice = DynamicMetrics.TEST_SQUARE_OVERSHOOT)

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
        """if self.lowest_test_small_entropy is not None:
            if not model.state_is_final:
                current_test_small_entropy = custom_metrics.get_test_overshoot_entropy_metric()
            else:
                current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
                self.equal_thresh = 0
            elif (current_test_small_entropy > self.lowest_test_small_entropy * 1.005 and not model.state_is_final) or (current_test_small_entropy > self.lowest_test_small_entropy * 1.02 and model.state_is_final):
                self.countdown += 1
                if self.countdown > 10:
                    if not model.state_is_final:
                        model.state_is_final = True
                        self.lowest_test_small_entropy = None
                        self.countdown = 0
                        self.equal_thresh = 0
                        return True
                    else:
                        return True
            elif self.lowest_test_small_entropy * 0.9995 < current_test_small_entropy and current_test_small_entropy < self.lowest_test_small_entropy * 1.005:
                self.equal_thresh += 1
                if self.equal_thresh > 20 and not model.state_is_final:
                    self.equal_thresh = 0
                    self.lowest_test_small_entropy = self.lowest_test_small_entropy * 0.9997

        else:
            if not model.state_is_final:
                current_test_small_entropy = custom_metrics.get_test_overshoot_entropy_metric()
            else:
                current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            self.lowest_test_small_entropy = current_test_small_entropy
        return False"""
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