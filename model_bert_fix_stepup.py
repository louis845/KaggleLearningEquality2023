import data_bert
import data_bert_tree_struct
import model_bert_fix
import tensorflow as tf
import numpy as np
import config
import gc
#

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size=512):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis=2)

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianDropout(rate=0.2)

        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        # the results of dense2 will be plugged into this.
        self.denseOvershoot = tf.keras.layers.Dense(units=128, activation="relu")
        self.dropoutOvershoot = tf.keras.layers.Dropout(rate=0.1)
        self.finalOvershoot = tf.keras.layers.Dense(units=1, activation="sigmoid")

        # dense1_fp takes in the combined input of dense0 and denseOvershoot
        self.dense1_fp = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout1_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense2_fp = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout2_fp = tf.keras.layers.Dropout(rate=0.1)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.dense4 = tf.keras.layers.Dense(units=128)
        self.relu4 = tf.keras.layers.ReLU()
        self.dropout4 = tf.keras.layers.Dropout(rate=0.1)
        self.dense5 = tf.keras.layers.Dense(units=1, activation="sigmoid")

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
    def call(self, data, training=False, actual_y=None):
        contents_description = data["contents"]["description"]
        contents_title = data["contents"]["title"]
        contents_lang = data["contents"]["lang"]
        topics_description = data["topics"]["description"]
        topics_title = data["topics"]["title"]
        topics_lang = data["topics"]["lang"]
        # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))
        embedding_result = self.concat_layer(
            [contents_description, contents_title, contents_lang, topics_description, topics_title, topics_lang])

        first_layer = self.dropout0(embedding_result, training=training)
        t = self.dropout1(self.dense1(first_layer), training=training)
        res_dropout2 = self.dropout2(self.dense2(t), training=training)

        overshoot_128result = self.dropoutOvershoot(self.denseOvershoot(res_dropout2), training=training)
        overshoot_result = self.finalOvershoot(overshoot_128result)


        t = self.dropout1_fp(self.dense1_fp(tf.concat([first_layer, overshoot_128result], axis=-1)), training=training)
        t = self.dropout2_fp(self.dense2_fp(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dropout4(self.relu4(self.dense4(t)), training=training)
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
    def compile(self):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0005, weight_decay=0.01)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000
        self.prev_entropy = None

        # saves the functions that generates whether a tuple (topic, content) is correlated.
        self.sample_generation_functions = {
            "train_sample": data_bert.obtain_train_sample,
            "test_sample": data_bert.obtain_test_sample,
            "train_square_sample": data_bert.obtain_train_square_sample,
            "test_square_sample": data_bert.obtain_test_square_sample
        }

        # generates the overshoot functions
        self.sample_overshoot_generation_functions = {
            "train_sample": data_bert_tree_struct.obtain_train_sample,
            "test_sample": data_bert_tree_struct.obtain_test_sample,
            "train_square_sample": data_bert_tree_struct.obtain_train_square_sample,
            "test_square_sample": data_bert_tree_struct.obtain_test_square_sample
        }
        # a function to test whether or not the tuple has correlations in the overshoot sense
        self.sample_overshoot_testing_function = data_bert_tree_struct.has_close_correlations

    # custom_generation_functions should be a dict, e.g. {"train_sample": data_bert.obtain_train_sample,
    #             "test_sample": data_bert.obtain_test_sample,
    #             "train_square_sample": data_bert.obtain_train_square_sample,
    #             "test_square_sample": data_bert.obtain_test_square_sample}
    def set_training_params(self, training_zero_sample_size=1000, training_one_sample_size=1000, training_max_size=None,
                            training_sampler=None, custom_metrics=None, custom_stopping_func=None,
                            custom_generation_functions=None, custom_overshoot_generation_functions=None, custom_overshoot_testing_function=None):
        self.training_one_sample_size = training_one_sample_size
        self.training_zero_sample_size = training_zero_sample_size
        self.training_max_size = training_max_size
        if training_sampler is not None:
            self.training_sampler = training_sampler
        if custom_metrics is not None:
            custom_metrics.set_training_sampler(self.training_sampler)
            self.custom_metrics = custom_metrics
        if custom_stopping_func is not None:
            self.custom_stopping_func = custom_stopping_func
        if custom_generation_functions is not None:
            self.sample_generation_functions = custom_generation_functions
        if custom_overshoot_generation_functions is not None:
            self.sample_overshoot_generation_functions = custom_overshoot_generation_functions
        if custom_overshoot_testing_function is not None:
            self.sample_overshoot_testing_function = custom_overshoot_testing_function

    def train_step(self, data):
        for k in range(50):
            # two pass, we first compute on overshoot only, and then compute on the full thing
            topics, contents, cors = self.sample_overshoot_generation_functions["train_sample"](
                one_sample_size=self.training_one_sample_size,
                zero_sample_size=self.training_zero_sample_size)
            input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
            cors = np.tile(cors, 2)
            y = tf.constant(cors)

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True)[:, 1]
                loss = self.loss(y, y_pred)
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            topics, contents, cors = self.sample_generation_functions["train_sample"](
                one_sample_size=self.training_one_sample_size,
                zero_sample_size=self.training_zero_sample_size)
            input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
            cors = np.tile(cors, 2)
            cors2 = np.tile(self.sample_overshoot_testing_function(contents, topics), 2)
            y = tf.constant(np.concatenate([np.expand_dims(cors, axis = 1), np.expand_dims(cors2, axis = 1)], axis=1))

            with tf.GradientTape() as tape:
                y_pred = self(input_data, training=True)
                loss = self.loss(y, y_pred)
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.training_max_size is None:
            limit = 9223372036854775807
            limit_sq = 9223372036854775807
        else:
            limit = self.training_max_size // 2
            limit_sq = int(np.sqrt(self.training_max_size))

        # evaluation at larger subset
        topics, contents, cors = self.sample_generation_functions["train_sample"](
            one_sample_size=self.training_one_sample_size,
            zero_sample_size=self.training_zero_sample_size)
        input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
        cors = np.tile(cors, 2)
        cors2 = np.tile(self.sample_overshoot_testing_function(contents, topics), 2)
        y = tf.constant(np.concatenate([np.expand_dims(cors, axis=1), np.expand_dims(cors2, axis=1)], axis=1))
        y_pred = self(input_data)
        self.entropy_large_set.update_state(y, y_pred)

        new_entropy = self.entropy_large_set.result()
        if (self.prev_entropy is not None) and new_entropy > self.prev_entropy * 1.05:
            print(
                "---------------------------------WARNING: Training problem: entropy has increased! Reverting training....---------------------------------")
            self.load_weights(config.training_models_path + "temp_ckpt/prev_ckpt")
        else:
            self.save_weights(config.training_models_path + "temp_ckpt/prev_ckpt")
            self.prev_entropy = new_entropy

        for m in self.metrics:
            m.update_state(y, y_pred)

        # eval other test metrics
        if self.custom_metrics is not None:
            self.custom_metrics.update_metrics(self, limit_sq, self.sample_generation_functions, self.sample_overshoot_generation_functions)

        # early stopping
        if (self.custom_stopping_func is not None) and self.custom_stopping_func.evaluate(self.custom_metrics, self):
            self.stop_training = True

        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics},
                "entropy_large_set": self.entropy_large_set.result(), **self.custom_metrics.obtain_metrics()}

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]

class DefaultMetrics(model_bert_fix.CustomMetrics):
    def __init__(self):
        model_bert_fix.CustomMetrics.__init__(self)
        threshold = 0.5

        self.test_precision = tf.keras.metrics.Precision(name="test_precision", thresholds=threshold)
        self.test_recall = tf.keras.metrics.Recall(name="test_recall", thresholds=threshold)

        self.test_small_precision = tf.keras.metrics.Precision(name="test_small_precision", thresholds=threshold)
        self.test_small_recall = tf.keras.metrics.Recall(name="test_small_recall", thresholds=threshold)
        self.test_small_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_small_accuracy", threshold=threshold)
        self.test_small_entropy = tf.keras.metrics.BinaryCrossentropy(name="test_small_entropy")

        self.no_lang_test_precision = tf.keras.metrics.Precision(name="no_lang_test_precision", thresholds=threshold)
        self.no_lang_test_recall = tf.keras.metrics.Recall(name="no_lang_test_recall", thresholds=threshold)

        self.no_lang_test_small_precision = tf.keras.metrics.Precision(name="no_lang_test_small_precision",
                                                                       thresholds=threshold)
        self.no_lang_test_small_recall = tf.keras.metrics.Recall(name="no_lang_test_small_recall", thresholds=threshold)
        self.no_lang_test_small_accuracy = tf.keras.metrics.BinaryAccuracy(name="no_lang_test_small_accuracy",
                                                                           threshold=threshold)
        self.no_lang_test_small_entropy = tf.keras.metrics.BinaryCrossentropy(name="no_lang_test_small_entropy")

        self.test_overshoot_precision = tf.keras.metrics.Precision(name="test_overshoot_precision", thresholds=threshold)
        self.test_overshoot_recall = tf.keras.metrics.Recall(name="test_overshoot_recall", thresholds=threshold)
        self.test_overshoot_small_precision = tf.keras.metrics.Precision(name="test_overshoot_small_precision", thresholds=threshold)
        self.test_overshoot_small_recall = tf.keras.metrics.Recall(name="test_overshoot_small_recall", thresholds=threshold)

    # updates the metrics based on the current state of the model.
    def update_metrics(self, model, limit_sq, sample_generation_functions, sample_overshoot_generation_functions):
        # evaluation at other points

        # test square sample
        topics, contents, cors = sample_generation_functions["test_square_sample"](min(600, limit_sq))
        input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 0]
        for m in self.test_metrics:
            m.update_state(y, y_pred)

        input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 0]
        for m in self.no_lang_test_metrics:
            m.update_state(y, y_pred)

        # test same sample
        topics, contents, cors = sample_generation_functions["test_sample"](30000, 30000)
        input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 0]
        for m in self.test_small_metrics:
            m.update_state(y, y_pred)

        input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 0]
        for m in self.no_lang_test_small_metrics:
            m.update_state(y, y_pred)

        # test square sample overshoot
        topics, contents, cors = sample_overshoot_generation_functions["test_square_sample"](min(600, limit_sq))
        input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 1]
        for m in self.test_overshoot_metrics:
            m.update_state(y, y_pred)

        # test same sample overshoot
        topics, contents, cors = sample_overshoot_generation_functions["test_sample"](30000, 30000)
        input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
        y_pred = model(input_data)[:, 1]
        for m in self.test_overshoot_small_metrics:
            m.update_state(y, y_pred)

        gc.collect()

    # returns a dictionary containing the last evaluation of the metrics, and the model
    def obtain_metrics(self):
        return {** {m.name: m.result() for m in self.test_metrics},
        ** {m.name: m.result() for m in self.test_small_metrics},
        ** {m.name: m.result() for m in self.no_lang_test_metrics},
        ** {m.name: m.result() for m in self.no_lang_test_small_metrics},
        **{m.name: m.result() for m in self.test_overshoot_metrics},
        **{m.name: m.result() for m in self.test_overshoot_small_metrics}}

    @property
    def test_metrics(self):
        return [self.test_precision, self.test_recall]

    @property
    def test_small_metrics(self):
        return [self.test_small_accuracy, self.test_small_precision, self.test_small_recall, self.test_small_entropy]

    @property
    def no_lang_test_metrics(self):
        return [self.no_lang_test_precision, self.no_lang_test_recall]

    @property
    def no_lang_test_small_metrics(self):
        return [self.no_lang_test_small_accuracy, self.no_lang_test_small_precision, self.no_lang_test_small_recall,
                self.no_lang_test_small_entropy]

    @property
    def test_overshoot_metrics(self):
        return [self.test_overshoot_precision, self.test_overshoot_recall]

    @property
    def test_overshoot_small_metrics(self):
        return [self.test_overshoot_small_precision, self.test_overshoot_small_recall]

class DefaultStoppingFunc(model_bert_fix.CustomStoppingFunc):
    def __init__(self, model_dir):
        model_bert_fix.CustomStoppingFunc.__init__(self, model_dir)
        self.lowest_test_small_entropy = None
        self.countdown = 0

    def evaluate(self, custom_metrics, model):
        if self.lowest_test_small_entropy is not None:
            current_test_small_entropy = custom_metrics.test_small_entropy.result()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
            elif current_test_small_entropy > self.lowest_test_small_entropy * 1.005:
                self.countdown += 1
                if self.countdown > 10:
                    return True
        else:
            current_test_small_entropy = custom_metrics.test_small_entropy.result()
            self.lowest_test_small_entropy = current_test_small_entropy
        return False