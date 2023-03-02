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
import tensorflow as tf
import numpy as np
import data_bert
import data_bert_tree_struct
import config
import gc

import data_bert_sampler


class TrainingSampler:
    # loads the data as tf tensors from the folders specified. note that the embedded_vectors_folder require "/" at the end.
    # for memory constrained GPU devices, we load the files into np arrays instead. in this case, it is not possible to
    # perform setwise operations.
    def __init__(self, embedded_vectors_folder, contents_one_hot_file, topics_one_hot_file, device = "gpu"):
        self.device = device
        if device == "gpu":
            self.contents_description = tf.constant(np.load(embedded_vectors_folder + "contents_description.npy"), dtype = tf.float32)
            self.contents_title = tf.constant(np.load(embedded_vectors_folder + "contents_title.npy"), dtype = tf.float32)
            self.topics_description = tf.constant(np.load(embedded_vectors_folder + "topics_description.npy"), dtype = tf.float32)
            self.topics_title = tf.constant(np.load(embedded_vectors_folder + "topics_title.npy"), dtype = tf.float32)

            self.contents_one_hot = tf.constant(np.load(contents_one_hot_file), dtype = tf.float32)
            self.topics_one_hot = tf.constant(np.load(topics_one_hot_file), dtype = tf.float32)
        else:
            self.contents_description = np.load(embedded_vectors_folder + "contents_description.npy")
            self.contents_title = np.load(embedded_vectors_folder + "contents_title.npy")
            self.topics_description = np.load(embedded_vectors_folder + "topics_description.npy")
            self.topics_title = np.load(embedded_vectors_folder + "topics_title.npy")

            self.contents_one_hot = np.load(contents_one_hot_file)
            self.topics_one_hot = np.load(topics_one_hot_file)

    def obtain_total_num_topics(self):
        return self.topics_one_hot.shape[0]

    def obtain_total_num_contents(self):
        return self.contents_one_hot.shape[0]

    # topics_id and contents_id are the indices for topics and contents. each entry in the batch is (topics_id[k], contents_id[k]).
    # the indices are in integer form (corresponding to data_bert.train_contents_num_id etc).
    def obtain_input_data(self, topics_id, contents_id):
        if self.device == "gpu":
            input_data = {
                "contents": {
                    "description": tf.gather(self.contents_description, np.expand_dims(contents_id, axis = 1), axis=0),
                    "title": tf.gather(self.contents_title, np.expand_dims(contents_id, axis = 1), axis=0),
                    "lang": tf.gather(self.contents_one_hot, np.expand_dims(contents_id, axis = 1), axis=0)
                },
                "topics": {
                    "description": tf.gather(self.topics_description, np.expand_dims(topics_id, axis = 1), axis=0),
                    "title": tf.gather(self.topics_title, np.expand_dims(topics_id, axis = 1), axis=0),
                    "lang": tf.gather(self.topics_one_hot, np.expand_dims(topics_id, axis = 1), axis=0)
                }
            }
        else:
            input_data = {
                "contents": {
                    "description": tf.constant(np.take(self.contents_description, np.expand_dims(contents_id, axis=1), axis=0), dtype = tf.float32),
                    "title": tf.constant(np.take(self.contents_title, np.expand_dims(contents_id, axis=1), axis=0), dtype = tf.float32),
                    "lang": tf.constant(np.take(self.contents_one_hot, np.expand_dims(contents_id, axis=1), axis=0), dtype = tf.float32)
                },
                "topics": {
                    "description": tf.constant(np.take(self.topics_description, np.expand_dims(topics_id, axis=1), axis=0), dtype = tf.float32),
                    "title": tf.constant(np.take(self.topics_title, np.expand_dims(topics_id, axis=1), axis=0), dtype = tf.float32),
                    "lang": tf.constant(np.take(self.topics_one_hot, np.expand_dims(topics_id, axis=1), axis=0), dtype = tf.float32)
                }
            }
        return input_data

    # same thing, except that the lang one-hot columns are all zeros.
    def obtain_input_data_filter_lang(self, topics_id, contents_id):
        if self.device == "gpu":
            input_data = {
                "contents": {
                    "description": tf.gather(self.contents_description, np.expand_dims(contents_id, axis = 1), axis=0),
                    "title": tf.gather(self.contents_title, np.expand_dims(contents_id, axis = 1), axis=0),
                    "lang": tf.constant(np.zeros(shape = (len(contents_id), 1, self.contents_one_hot.shape[1])), dtype = tf.float32)
                },
                "topics": {
                    "description": tf.gather(self.topics_description, np.expand_dims(topics_id, axis = 1), axis=0),
                    "title": tf.gather(self.topics_title, np.expand_dims(topics_id, axis = 1), axis=0),
                    "lang": tf.constant(np.zeros(shape = (len(topics_id), 1, self.contents_one_hot.shape[1])), dtype = tf.float32)
                }
            }
        else:
            input_data = {
                "contents": {
                    "description": tf.constant(np.take(self.contents_description, np.expand_dims(contents_id, axis=1), axis=0), dtype = tf.float32),
                    "title": tf.constant(np.take(self.contents_title, np.expand_dims(contents_id, axis=1), axis=0), dtype = tf.float32),
                    "lang": tf.constant(np.zeros(shape=(len(contents_id), 1, self.contents_one_hot.shape[1])), dtype = tf.float32)
                },
                "topics": {
                    "description": tf.constant(np.take(self.topics_description, np.expand_dims(topics_id, axis=1), axis=0), dtype = tf.float32),
                    "title": tf.constant(np.take(self.topics_title, np.expand_dims(topics_id, axis=1), axis=0), dtype = tf.float32),
                    "lang": tf.constant(np.zeros(shape=(len(topics_id), 1, self.contents_one_hot.shape[1])), dtype = tf.float32)
                }
            }
        return input_data

    # obtain the version where both input data with lang and input data without lang exist.
    def obtain_input_data_both(self, topics_id, contents_id):
        with_lang = self.obtain_input_data(topics_id, contents_id)
        no_lang = self.obtain_input_data_filter_lang(topics_id, contents_id)

        input_data = {
            "contents": {
                "description": tf.concat([with_lang["contents"]["description"], no_lang["contents"]["description"]], axis = 0),
                "title": tf.concat([with_lang["contents"]["title"], no_lang["contents"]["title"]], axis = 0),
                "lang": tf.concat([with_lang["contents"]["lang"], no_lang["contents"]["lang"]], axis = 0)
            },
            "topics": {
                "description": tf.concat([with_lang["topics"]["description"], no_lang["topics"]["description"]], axis = 0),
                "title": tf.concat([with_lang["topics"]["title"], no_lang["topics"]["title"]], axis = 0),
                "lang": tf.concat([with_lang["topics"]["lang"], no_lang["topics"]["lang"]], axis = 0)
            }
        }

        del with_lang, no_lang
        return input_data

    # expected: 3 1D arrays of the same length. each tuple (topics_id_klevel[j], contents_id[j], tree_levels[j])
    # represents a topic subtree, and the content id. the content id is obviously contents_id[j]. The topic subtree
    # is given by data_bert_tree_struct.topics_group_filtered[tree_levels[j]]["group_filter_available"][topics_id_klevel[j]].
    # The topic_num_id of the root node of the subtree is
    # data_bert_tree_struct.topics_group_filtered[tree_levels[j]]["group_ids"][topics_id_klevel[j]]
    def obtain_input_data_tree(self, topics_id_klevel, contents_id, tree_levels, final_level):
        if self.device != "gpu":
            raise Exception("Must use GPU for tree learning!")
        assert len(topics_id_klevel) == len(contents_id) and len(contents_id) == len(tree_levels)

        topics_ragged_list = np.empty(shape=(len(topics_id_klevel)), dtype="object")
        for level in range(np.min(tree_levels), np.max(tree_levels) + 1):
            llocs = tree_levels == level
            if level == final_level: # last level is usual data
                topics_ragged_list[llocs] = data_bert_tree_struct.dummy_topics_prod_list[topics_id_klevel[llocs]]
            else:
                topics_ragged_list[llocs] = data_bert_tree_struct.topics_group_filtered[level]["group_filter_available"][topics_id_klevel[llocs]]
        topics_ragged_list = list(topics_ragged_list)
        contents_ragged_list = [np.repeat(contents_id[k], len(topics_ragged_list[k])) for k in range(len(topics_ragged_list))]

        topics_ragged_list = tf.ragged.constant(topics_ragged_list)
        contents_ragged_list = tf.ragged.constant(contents_ragged_list)
        input_data = {
            "contents": {
                "description": tf.gather(self.contents_description, contents_ragged_list, axis=0),
                "title": tf.gather(self.contents_title, contents_ragged_list, axis=0),
                "lang": tf.gather(self.contents_one_hot, contents_ragged_list, axis=0)
            },
            "topics": {
                "description": tf.gather(self.topics_description, topics_ragged_list, axis=0),
                "title": tf.gather(self.topics_title, topics_ragged_list, axis=0),
                "lang": tf.gather(self.topics_one_hot, topics_ragged_list, axis=0)
            }
        }
        return input_data

    # same thing, except that the lang one-hot columns are all zeros.
    def obtain_input_data_tree_filter_lang(self, topics_id_klevel, contents_id, tree_levels, final_level):
        if self.device != "gpu":
            raise Exception("Must use GPU for tree learning!")
        assert len(topics_id_klevel) == len(contents_id) and len(contents_id) == len(tree_levels)

        topics_ragged_list = np.empty(shape=(len(topics_id_klevel)), dtype="object")
        for level in range(np.min(tree_levels), np.max(tree_levels) + 1):
            llocs = tree_levels == level
            if level == final_level:  # last level is usual data
                topics_ragged_list[llocs] = data_bert_tree_struct.dummy_topics_prod_list[topics_id_klevel[llocs]]
            else:
                topics_ragged_list[llocs] = data_bert_tree_struct.topics_group_filtered[level]["group_filter_available"][topics_id_klevel[llocs]]
        topics_ragged_list = list(topics_ragged_list)
        contents_ragged_list = [np.repeat(contents_id[k], len(topics_ragged_list[k])) for k in
                                range(len(topics_ragged_list))]
        dummy_ragged_list = [np.repeat(0, len(topics_ragged_list[k])) for k in
                                range(len(topics_ragged_list))]

        topics_ragged_list = tf.ragged.constant(topics_ragged_list)
        contents_ragged_list = tf.ragged.constant(contents_ragged_list)
        dummy_ragged_list = tf.ragged.constant(dummy_ragged_list)

        dummy_zeros = tf.zeros(shape = (1, self.topics_one_hot.shape[1]), dtype = tf.float32)
        input_data = {
            "contents": {
                "description": tf.gather(self.contents_description, contents_ragged_list, axis=0),
                "title": tf.gather(self.contents_title, contents_ragged_list, axis=0),
                "lang": tf.gather(dummy_zeros, dummy_ragged_list, axis=0)
            },
            "topics": {
                "description": tf.gather(self.topics_description, topics_ragged_list, axis=0),
                "title": tf.gather(self.topics_title, topics_ragged_list, axis=0),
                "lang": tf.gather(dummy_zeros, dummy_ragged_list, axis=0)
            }
        }
        return input_data

    def obtain_input_data_tree_both(self, topics_id_klevel, contents_id, tree_levels, final_level):
        with_lang = self.obtain_input_data_tree(topics_id_klevel, contents_id, tree_levels, final_level)
        no_lang = self.obtain_input_data_tree_filter_lang(topics_id_klevel, contents_id, tree_levels, final_level)

        input_data = {
            "contents": {
                "description": tf.concat([with_lang["contents"]["description"], no_lang["contents"]["description"]], axis = 0),
                "title": tf.concat([with_lang["contents"]["title"], no_lang["contents"]["title"]], axis = 0),
                "lang": tf.concat([with_lang["contents"]["lang"], no_lang["contents"]["lang"]], axis = 0)
            },
            "topics": {
                "description": tf.concat([with_lang["topics"]["description"], no_lang["topics"]["description"]], axis = 0),
                "title": tf.concat([with_lang["topics"]["title"], no_lang["topics"]["title"]], axis = 0),
                "lang": tf.concat([with_lang["topics"]["lang"], no_lang["topics"]["lang"]], axis = 0)
            }
        }

        del with_lang, no_lang
        return input_data

    def __del__(self):
        del self.contents_title, self.contents_description, self.contents_one_hot, self.topics_title, self.topics_description, self.topics_one_hot
        gc.collect()
class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size = 512, init_noise = 0.05):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis = 2)

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianNoise(stddev = init_noise)
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.dense4 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense4")
        self.dropout4 = tf.keras.layers.Dropout(rate=0.1)
        self.dense5 = tf.keras.layers.Dense(units=units_size // 4, activation="relu", name="dense5")
        self.dropout5 = tf.keras.layers.Dropout(rate=0.1)
        self.dense_final = tf.keras.layers.Dense(units=1, activation="sigmoid", name = "dense_final")


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
    def compile(self, weight_decay = 0.01):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0005, weight_decay = weight_decay)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000
        self.prev_entropy = None

        self.tuple_choice_sampler = data_bert_sampler.default_sampler_instance
    def set_training_params(self, training_sample_size = 15000, training_max_size = None, training_sampler = None, custom_metrics = None, custom_stopping_func = None, custom_tuple_choice_sampler = None):
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
    # for tree learning, final_tree_level is a boolean mask to indicate which batches are
    # in the final level (individual level), which batches are tree batches.
    def call(self, data, training=False, actual_y = None, final_tree_level = None):
        contents_description = data["contents"]["description"]
        contents_title = data["contents"]["title"]
        contents_lang = data["contents"]["lang"]
        topics_description = data["topics"]["description"]
        topics_title = data["topics"]["title"]
        topics_lang = data["topics"]["lang"]
        # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))
        embedding_result = self.concat_layer([contents_description, contents_title, contents_lang, topics_description, topics_title, topics_lang])

        shape = embedding_result.shape
        if len([1 for k in range(int(shape.rank)) if shape[k] is None]) > 0:
            t = tf.ragged.map_flat_values(self.dropout0, embedding_result, training=training)
        else:
            t = self.dropout0(embedding_result, training=training)
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dropout4(self.dense4(t), training=training)
        t = self.dropout5(self.dense5(t), training=training)
        t = self.dense_final(t) # now we have a batch_size x set_size x 1 tensor, the last axis is reduced to 1 by linear transforms.
        if (training and actual_y is not None and actual_y.shape[0] is not None
            and actual_y.shape[0] == shape[0] and final_tree_level is not None and
                final_tree_level.shape[0] is not None and final_tree_level.shape[0] == shape[0]): # here we use overestimation training method for the set
            p = 4.0
            tclip = tf.clip_by_value(t, clip_value_min=0.05, clip_value_max=0.95)
            pmean = tf.math.pow(tclip, p)
            pmean = tf.reduce_mean(pmean, axis = 1)
            pmean = tf.squeeze(tf.math.pow(pmean, 1 / p), axis = 1)

            pinvmean = tf.math.pow(tclip, p)
            pinvmean = tf.reduce_mean(pinvmean, axis=1)
            pinvmean = tf.clip_by_value(tf.squeeze(tf.math.pow(pinvmean, 1 / p), axis = 1),
                                        clip_value_min=0.05, clip_value_max=0.55)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.add(pinvmean * tf.constant(actual_y, dtype = tf.float32),
                                pmean * tf.constant(1 - actual_y, dtype = tf.float32))

            proba = tf.math.add(tf.squeeze(tf.reduce_max(t, axis = 1), axis = 1) * tf.constant(final_tree_level, dtype=tf.float32),
                                proba * tf.constant(1 - final_tree_level, dtype=tf.float32))
            return proba
        else: # here we just return the probabilities normally. the probability will be computed as the max inside the set
            return tf.squeeze(tf.reduce_max(t, axis = 1), axis = 1)

    def eval_omit_last(self, data, training=False):
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
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        return self.dense4(t) # now we have a batch_size x set_size x 128 tensor, the last axis is reduced to 128 by linear transforms.

    def train_step_tree(self):
        for k in range(50):
            topics, contents, cors, class_ids, tree_levels, multipliers = self.tuple_choice_sampler.obtain_train_sample(self.training_sample_size)
            input_data = self.training_sampler.obtain_input_data_tree_both(topics, contents, tree_levels, 5)
            cors = np.tile(cors, 2)
            y = tf.expand_dims(tf.constant(cors, dtype = tf.float32), axis = 1)
            multipliers_tf = tf.constant(np.tile(multipliers, 2), dtype = tf.float32)
            final_tree_level = np.tile((tree_levels == 5).astype(dtype=np.float32), 2)

            with tf.GradientTape() as tape:
                y_pred = tf.expand_dims(self(input_data, actual_y = cors, training=True, final_tree_level = final_tree_level), axis = 1)
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
        input_data = self.training_sampler.obtain_input_data_tree_both(topics, contents, tree_levels, 5)
        cors = np.tile(cors, 2)
        y = tf.expand_dims(tf.constant(cors, dtype = tf.float32), axis=1)
        multipliers_tf = tf.constant(np.tile(multipliers, 2), dtype = tf.float32)
        final_tree_level = np.tile((tree_levels == 5).astype(dtype=np.float32), 2)

        y_pred = tf.expand_dims(self(input_data, actual_y = cors, training=True, final_tree_level=final_tree_level), axis = 1)
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
            input_data = self.training_sampler.obtain_input_data_both(topics_id = topics, contents_id = contents)
            cors = np.tile(cors, 2)
            y = tf.constant(cors, dtype = tf.float32)

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
        topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(min(len(data_bert.train_contents), limit))
        cors = np.tile(cors, 2)
        input_data = self.training_sampler.obtain_input_data_both(topics_id=topics, contents_id=contents)
        y = tf.constant(cors, dtype = tf.float32)
        y_pred = self(input_data)
        self.entropy_large_set.update_state(y, y_pred)

        new_entropy = self.entropy_large_set.result()
        if (self.prev_entropy is not None) and new_entropy > self.prev_entropy * 1.05:
            print("---------------------------------WARNING: Training problem: entropy has increased! Reverting training....---------------------------------")
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

class CustomMetrics:
    def __init__(self):
        self.training_sampler = None

    # updates the metrics based on the current state of the model. limit_sq is the limit of the square size.
    # sample_generation_functions are the dict of 4 functions for the train set, train square set, test set, test square set etc.
    def update_metrics(self, model, sample_size_limit):
        pass

    # returns a dictionary containing the last evaluation of the metrics, and the model
    def obtain_metrics(self):
        pass

    def set_training_sampler(self, training_sampler):
        self.training_sampler = training_sampler

class DynamicMetrics(CustomMetrics):

    TRAIN = 1
    TRAIN_SQUARE = 2
    TEST = 3
    TEST_SQUARE = 4

    def __init__(self):
        CustomMetrics.__init__(self)
        self.metrics = [] # a lists of dicts, containing the metrics, and the data_bert_sampler.SamplerBase which contains the metric
        self.tree_metrics = [] # similar.

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

    def add_tree_metric(self, name, tuple_choice_sampler_tree, level, sample_choice = TEST, threshold = 0.6):
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
            sample_choice = self.metrics[k]["sample_choice"]

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
            y_pred = model(input_data)
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_filter_lang(topics_id=topics, contents_id=contents)
            y_pred = model(input_data)
            for j in range(4,8):
                kmetrics[j].update_state(y, y_pred)

        for k in range(len(self.tree_metrics)):
            kmetrics = self.tree_metrics[k]["metrics"]
            sampler = self.tree_metrics[k]["sampler"]
            sample_choice = self.tree_metrics[k]["sample_choice"]
            level = self.tree_metrics[k]["level"]

            if sample_choice == DynamicMetrics.TRAIN:
                topics, contents, cors = sampler.obtain_tree_train_sample(min(50, sample_size_limit), level)
            elif sample_choice == DynamicMetrics.TRAIN_SQUARE:
                topics, contents, cors = sampler.obtain_tree_train_square_sample(min(7, sample_size_limit), level)
            elif sample_choice == DynamicMetrics.TEST:
                topics, contents, cors = sampler.obtain_tree_test_sample(min(50, sample_size_limit), level)
            elif sample_choice == DynamicMetrics.TEST_SQUARE:
                topics, contents, cors = sampler.obtain_tree_test_square_sample(min(7, sample_size_limit), level)
            input_data = self.training_sampler.obtain_input_data_tree(topics_id_klevel=topics, contents_id=contents,
                                                                      tree_levels=np.repeat(level, len(cors)),final_level=5)
            y = tf.constant(cors, dtype = tf.float32)
            y_pred = model(input_data)
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)

            input_data = self.training_sampler.obtain_input_data_tree_filter_lang(topics_id_klevel=topics, contents_id=contents,
                                                                      tree_levels=np.repeat(level, len(cors)),final_level=5)
            y_pred = model(input_data)
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

    def get_total_tree_metric(self):
        sum = 0.0
        for k in range(len(self.tree_metrics)):
            mmetrics = self.tree_metrics[k]["metrics"]
            sum += mmetrics[3].result()
        return sum + self.get_test_entropy_metric()

default_metrics = DynamicMetrics()
default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST)
default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST_SQUARE)

default_tree_metrics = DynamicMetrics()
default_tree_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST_SQUARE)
default_tree_metrics.add_tree_metric("treelv0", data_bert_sampler.default_tree_sampler_instance, level = 0, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv1", data_bert_sampler.default_tree_sampler_instance, level = 1, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv2", data_bert_sampler.default_tree_sampler_instance, level = 2, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv3", data_bert_sampler.default_tree_sampler_instance, level = 3, sample_choice = DynamicMetrics.TEST)
default_tree_metrics.add_tree_metric("treelv4", data_bert_sampler.default_tree_sampler_instance, level = 4, sample_choice = DynamicMetrics.TEST)

# create overshoot metrics, given the sampler used for selecting the tuples
def obtain_overshoot_metric_instance(training_tuple_sampler):
    overshoot_metrics = DynamicMetrics()
    overshoot_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, sample_choice=DynamicMetrics.TEST_SQUARE)
    overshoot_metrics.add_metric("test_in_train_sample", training_tuple_sampler, sample_choice=DynamicMetrics.TEST)
    overshoot_metrics.add_metric("test_square_in_train_sample", training_tuple_sampler, sample_choice=DynamicMetrics.TEST_SQUARE)
    return overshoot_metrics

class CustomStoppingFunc:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    # Returns true if the model needs to stop, otherwise False
    def evaluate(self, custom_metrics, model):
        return False

class DefaultStoppingFunc(CustomStoppingFunc):
    def __init__(self, model_dir):
        CustomStoppingFunc.__init__(self, model_dir)
        self.lowest_test_small_entropy = None
        self.countdown = 0

    def evaluate(self, custom_metrics, model):
        if self.lowest_test_small_entropy is not None:
            current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
            elif current_test_small_entropy > self.lowest_test_small_entropy * 1.005:
                self.countdown += 1
                if self.countdown > 10:
                    return True
        else:
            current_test_small_entropy = custom_metrics.get_test_entropy_metric()
            self.lowest_test_small_entropy = current_test_small_entropy
        return False

class DefaultTreeStoppingFunc(CustomStoppingFunc):
    def __init__(self, model_dir):
        CustomStoppingFunc.__init__(self, model_dir)
        self.lowest_test_small_entropy = None
        self.countdown = 0

    def evaluate(self, custom_metrics, model):
        if self.lowest_test_small_entropy is not None:
            current_test_small_entropy = custom_metrics.get_total_tree_metric()
            if current_test_small_entropy < self.lowest_test_small_entropy:
                self.lowest_test_small_entropy = current_test_small_entropy
                model.save_weights(self.model_dir + "/best_test_small_entropy.ckpt")
                self.countdown = 0
            elif current_test_small_entropy > self.lowest_test_small_entropy * 1.005:
                self.countdown += 1
                if self.countdown > 10:
                    return True
        else:
            current_test_small_entropy = custom_metrics.get_total_tree_metric()
            self.lowest_test_small_entropy = current_test_small_entropy
        return False