"""
A model for classifying topic-content correlations.

Model input: dimensions (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs)) (ragged tensor).

The 0th axis is the batch size. The 1st axis is the set_size for the current input, which may vary for each different instance in the
batch. The 2nd axis are the precomputed inputs from BERT embedding concatenated with the one-hot representation of languages. The order
is (content_description, content_title, content_lang, topic_description, topic_title, topic_lang).

Each input [k,:,:] denotes the kth input to model, which is a set of topics and contents tuples. The content must be the same throughout
the set [k,:,:]. Note [k,j,:] denotes the tuple (content_description, content_title, content_lang, topic_description, topic_title, topic_lang)
belonging to the jth element in the set of kth input to the model.

The actual input is input=dict, where input["contents"]["description"], input["contents"]["title"], input["contents"]["lang"]
are the tensors with same batch size, and each [k,:,:] size corresponds to the same single sample, and each [k,j,:] denotes the same
tuple in the same sample.

Model output:
A (batch_size) tensor (vector) containing the predicted probabilities. The model tries to predict whether the set of topics contain the
given content.
"""
import tensorflow
import tensorflow as tf
import numpy as np
import data_bert
import data_bert_tree_struct
import config
import gc

import data_bert_sampler
import model_bert_fix


class Model(tf.keras.Model):
    # trivial model.
    def __init__(self):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis=2)

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianDropout(rate=0.2)
        self.dense = tf.keras.layers.Dense(units=1, activation="sigmoid", name="dense")

        # loss functions and eval metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision = tf.keras.metrics.Precision(name="precision")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.entropy = tf.keras.metrics.BinaryCrossentropy(name="entropy")
        self.entropy_large_set = tf.keras.metrics.BinaryCrossentropy(name="entropy_large_set")

        # metrics for test set
        self.custom_metrics = None
        self.custom_stopping_func = None

        self.tuple_choice_sampler = None

    def compile(self, weight_decay=0.01):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0005, weight_decay=weight_decay)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000
        self.prev_entropy = None

        self.tuple_choice_sampler = data_bert_sampler.default_sampler_instance

    def set_training_params(self, training_sample_size=15000, training_max_size=None, training_sampler=None,
                            custom_metrics=None, custom_stopping_func=None, custom_tuple_choice_sampler=None):
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

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size) numpy vector.
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

        t = self.dropout0(embedding_result, training=training)
        t = self.dense(t)
        if training and actual_y is not None:  # here we use overestimation training method for the set
            p = 4.0
            pmean = tf.math.pow(t, p)
            pmean = tf.reduce_mean(pmean, axis=1)
            pmean = tf.squeeze(tf.math.pow(pmean, 1 / p), axis=1)

            pinvmean = tf.math.pow(t, 1 / p)
            pinvmean = tf.reduce_mean(pinvmean, axis=1)
            pinvmean = tf.squeeze(tf.math.pow(pinvmean, p), axis=1)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.add(pinvmean * tf.constant(actual_y), pmean * tf.constant(1 - actual_y))
            return proba
        else:  # here we just return the probabilities normally. the probability will be computed as the max inside the set
            return tf.squeeze(tf.reduce_max(t, axis=1), axis=1)
    def train_step_tree(self):
        for k in range(50):
            topics, contents, cors, class_ids, tree_levels, multipliers = self.tuple_choice_sampler.obtain_train_sample(
                self.training_sample_size)
            input_data = self.training_sampler.obtain_input_data_tree_both(topics, contents, tree_levels)
            cors = np.tile(cors, 2)
            y = tf.expand_dims(tf.constant(cors), axis=1)
            multipliers_tf = tf.constant(np.tile(multipliers, 2))

            with tf.GradientTape() as tape:
                y_pred = tf.expand_dims(self(input_data, actual_y=cors, training=True), axis=1)
                loss = self.loss(y, y_pred, sample_weight=multipliers_tf)

            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.training_max_size is None:
            limit = 9223372036854775807
            limit_sq = 9223372036854775807
        else:
            limit = self.training_max_size

        # evaluation at larger subset
        topics, contents, cors, class_ids, tree_levels, multipliers = self.tuple_choice_sampler.obtain_train_sample(
            self.training_sample_size)
        input_data = self.training_sampler.obtain_input_data_tree_both(topics, contents, tree_levels)
        cors = np.tile(cors, 2)
        y = tf.expand_dims(tf.constant(cors), axis=1)
        multipliers_tf = tf.constant(np.tile(multipliers, 2))
        y_pred = tf.expand_dims(self(input_data, actual_y=cors, training=True), axis=1)
        self.entropy_large_set.update_state(y, y_pred, sample_weight=multipliers_tf)

        new_entropy = self.entropy_large_set.result()
        if (self.prev_entropy is not None) and new_entropy > self.prev_entropy * 1.05:
            print(
                "---------------------------------WARNING: Training problem: entropy has increased! Reverting training....---------------------------------")
            # self.load_weights(config.training_models_path + "temp_ckpt/prev_ckpt")
        else:
            pass
            # self.save_weights(config.training_models_path + "temp_ckpt/prev_ckpt")
            # self.prev_entropy = new_entropy

        for m in self.metrics:
            m.update_state(y, y_pred)

        # eval other test metrics
        if self.custom_metrics is not None:
            self.custom_metrics.update_metrics(self, limit)

        # early stopping
        if (self.custom_stopping_func is not None) and self.custom_stopping_func.evaluate(self.custom_metrics, self):
            self.stop_training = True

        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics},
                "entropy_large_set": self.entropy_large_set.result(), **self.custom_metrics.obtain_metrics()}

    def train_step(self, data):
        if self.tuple_choice_sampler.is_tree_sampler():
            return self.train_step_tree()
        for k in range(50):
            topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(self.training_sample_size)
            input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
            cors = np.tile(cors, 2)
            y = tf.constant(cors)

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
            limit = self.training_max_size

        # evaluation at larger subset
        topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(
            min(len(data_bert.train_contents), limit))
        cors = np.tile(cors, 2)
        input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
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
            self.custom_metrics.update_metrics(self, limit)

        # early stopping
        if (self.custom_stopping_func is not None) and self.custom_stopping_func.evaluate(self.custom_metrics, self):
            self.stop_training = True

        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics},
                "entropy_large_set": self.entropy_large_set.result(), **self.custom_metrics.obtain_metrics()}

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]