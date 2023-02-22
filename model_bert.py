"""
A model for classifying topic-content correlations, WITH the BERT tokens as input. The BERT weights will be learnt alongside
the methods used in model_bert_fix

Model input: dimensions (batch_size x set_size x (bert_information)) (ragged tensor).

The 0th axis is the batch size. The 1st axis is the set_size for the current input, which may vary for each different instance in the
batch. The 2nd axis are the precomputed inputs from BERT embedding concatenated with the one-hot representation of languages. The order
is (content_description, content_title, content_lang, topic_description, topic_title, topic_lang).

Each input [k,:,:] denotes the kth input to model, which is a set of topics and contents tuples. The content must be the same throughout
the set [k,:,:]. Note [k,j,:] denotes the tuple (content_description, content_title, content_lang, topic_description, topic_title, topic_lang)
belonging to the jth element in the set of kth input to the model.

The actual input is input=dict, where input["contents"]["description"], input["contents"]["title"], input["contents"]["lang"]
are the tensors with same batch size, and each [k,:,:] size corresponds to the same single sample, and each [k,j,:] denotes the same
tuple in the same sample.

This model does not use the languages one hot information, to attain larger generalizability.

Model output:
A (batch_size) tensor (vector) containing the predicted probabilities. The model tries to predict whether the set of topics contain the
given content.
"""
import tensorflow
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import numpy as np
import data_bert
import config
import gc

import data_bert_sampler


class TrainingSampler:
    # loads the data tokens as tf tensors from the folders specified. note that the embedded_vectors_folder require "/" at the end.
    # for memory constrained GPU devices, we load the files into np arrays instead. in this case, it is not possible to
    # perform setwise operations.
    def __init__(self, tokens_folder, device = "gpu"):
        self.device = device
        self.model_input_params = ["input_mask", "input_type_ids", "input_word_ids"]
        self.contents_description = {}
        self.contents_title = {}
        self.topics_description = {}
        self.topics_title = {}
        if device == "gpu":
            for model_input in self.model_input_params:
                self.contents_description[model_input] = tf.constant(np.load(tokens_folder + "contents_description/" + model_input + ".npy"))
                self.contents_title[model_input] = tf.constant(np.load(tokens_folder + "contents_title/" + model_input + ".npy"))
                self.topics_description[model_input] = tf.constant(np.load(tokens_folder + "topics_description/" + model_input + ".npy"))
                self.topics_title[model_input] = tf.constant(np.load(tokens_folder + "topics_title/" + model_input + ".npy"))
        else:
            for model_input in self.model_input_params:
                self.contents_description[model_input] = np.load(tokens_folder + "contents_description/" + model_input + ".npy")
                self.contents_title[model_input] = np.load(tokens_folder + "contents_title/" + model_input + ".npy")
                self.topics_description[model_input] = np.load(tokens_folder + "topics_description/" + model_input + ".npy")
                self.topics_title[model_input] = np.load(tokens_folder + "topics_title/" + model_input + ".npy")

    def obtain_total_num_topics(self):
        return self.contents_description[self.model_input_params[0]].shape[0]

    def obtain_total_num_contents(self):
        return self.contents_description[self.model_input_params[0]].shape[0]

    # topics_id and contents_id are the indices for topics and contents. each entry in the batch is (topics_id[k], contents_id[k]).
    # the indices are in integer form (corresponding to data_bert.train_contents_num_id etc).
    def obtain_input_data(self, topics_id, contents_id):
        input_data = {
            "contents": {
                "description": {},
                "title": {},
            },
            "topics": {
                "description": {},
                "title": {},
            }
        }
        if self.device == "gpu":
            for model_input in self.model_input_params:
                input_data["contents"]["description"][model_input] = tf.gather(self.contents_description[model_input], contents_id, axis=0)
                input_data["contents"]["title"][model_input] = tf.gather(self.contents_title[model_input], contents_id, axis=0)
                input_data["topics"]["description"][model_input] = tf.gather(self.topics_description[model_input], topics_id, axis=0)
                input_data["topics"]["title"][model_input] = tf.gather(self.topics_title[model_input], contents_id, axis=0)
        else:
            for model_input in self.model_input_params:
                input_data["contents"]["description"][model_input] = tf.constant(self.contents_description[model_input][contents_id, :])
                input_data["contents"]["title"][model_input] = tf.constant(self.contents_title[model_input][contents_id, :])
                input_data["topics"]["description"][model_input] = tf.constant(self.topics_description[model_input][topics_id, :])
                input_data["topics"]["title"][model_input] = tf.constant(self.topics_title[model_input][topics_id, :])
        return input_data

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size = 512):
        super(Model, self).__init__()

        self.training_sampler = None
        self.training_max_size = None

        # BERT layer.
        self.bert_encoder_L12_H256 = hub.KerasLayer(
            "/kaggle/input/kagglelearningequalitybertmodels/small_bert_bert_en_uncased_L-12_H-256_A-4_2", trainable = True)

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis = 2)

        # standard stuff
        self.dropout0 = tf.keras.layers.GaussianDropout(rate=0.2)
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense1")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense2")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu", name = "dense3")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.1)
        self.dense4 = tf.keras.layers.Dense(units= (units_size // 4), name = "dense4")
        self.relu4 = tf.keras.layers.ReLU()
        self.dropout4 = tf.keras.layers.Dropout(rate=0.1)
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

        self.tuple_choice_sampler = None
    def compile(self, weight_decay = 0.01, learning_rate = 0.0005):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay = weight_decay)
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
    def call(self, data, training=False, actual_y = None):
        contents_description = data["contents"]["description"]
        contents_title = data["contents"]["title"]
        topics_description = data["topics"]["description"]
        topics_title = data["topics"]["title"]

        # feed them into the BERT
        contents_description = self.bert_encoder_L12_H256(contents_description)["pooled_output"]
        contents_title = self.bert_encoder_L12_H256(contents_title)["pooled_output"]
        topics_description = self.bert_encoder_L12_H256(topics_description)["pooled_output"]
        topics_title = self.bert_encoder_L12_H256(topics_title)["pooled_output"]

        # now it would be an n x bert_embedding_size tensor. we expand it.
        contents_description = tf.expand_dims(contents_description, axis=1)
        contents_title = tf.expand_dims(contents_title, axis=1)
        topics_description = tf.expand_dims(topics_description, axis=1)
        topics_title = tf.expand_dims(topics_title, axis=1)

        # combine the (batch_size x (bert_embedding_size / num_langs)) tensors into (batch_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))
        embedding_result = self.concat_layer([contents_description, contents_title, topics_description, topics_title])

        t = self.dropout0(embedding_result, training=training)
        t = self.dropout1(self.dense1(t), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dropout4(self.relu4(self.dense4(t)), training=training)
        t = self.dense5(t) # now we have a batch_size x set_size x 1 tensor, the last axis is reduced to 1 by linear transforms.
        if training and actual_y is not None: # here we use overestimation training method for the set
            p = 4.0
            pmean = tf.math.pow(t, p)
            pmean = tf.reduce_mean(pmean, axis = 1)
            pmean = tf.squeeze(tf.math.pow(pmean, 1 / p), axis = 1)

            pinvmean = tf.math.pow(t, 1 / p)
            pinvmean = tf.reduce_mean(pinvmean, axis=1)
            pinvmean = tf.squeeze(tf.math.pow(pinvmean, p), axis = 1)
            # note that pmean and pinvmean are "close" to max, harmonic mean respectively.
            # if actual_y is 1 we use the pinvmean, to encourage low prob topics to move
            # close to 1. if actual_y is 0 we use pmean, to encourage high prob topics to
            # move close to 0
            proba = tf.math.add(pinvmean * tf.constant(actual_y), pmean * tf.constant(1 - actual_y))
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
    def train_step(self, data):
        for k in range(50):
            topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(self.training_sample_size)
            input_data = self.training_sampler.obtain_input_data(topics_id = topics, contents_id = contents)
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
        topics, contents, cors, class_ids = self.tuple_choice_sampler.obtain_train_sample(min(len(data_bert.train_contents), limit))
        input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
        y = tf.constant(cors)
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

    def add_metric(self, name, tuple_choice_sampler, sample_choice = TEST, threshold = 0.5):
        accuracy = tf.keras.metrics.BinaryAccuracy(name = name + "_accuracy", threshold=threshold)
        precision = tf.keras.metrics.Precision(name = name + "_precision", thresholds=threshold)
        recall = tf.keras.metrics.Recall(name = name + "_recall", thresholds=threshold)
        entropy = tf.keras.metrics.BinaryCrossentropy(name = name + "_entropy")
        self.metrics.append({"metrics": [accuracy, precision, recall, entropy], "sampler": tuple_choice_sampler, "sample_choice": sample_choice})

    def update_metrics(self, model, sample_size_limit):
        for k in range(len(self.metrics)):
            kmetrics = self.metrics[k]["metrics"]
            sampler = self.metrics[k]["sampler"]
            sample_choice = self.metrics[k]["sample_choice"]

            if sample_choice == DynamicMetrics.TRAIN:
                topics, contents, cors, class_id = sampler.obtain_test_sample(min(60000, sample_size_limit))
            elif sample_choice == DynamicMetrics.TRAIN_SQUARE:
                topics, contents, cors, class_id = sampler.obtain_train_sample(min(360000, sample_size_limit))
            elif sample_choice == DynamicMetrics.TEST:
                topics, contents, cors, class_id = sampler.obtain_test_sample(min(60000, sample_size_limit))
            elif sample_choice == DynamicMetrics.TEST_SQUARE:
                topics, contents, cors, class_id = sampler.obtain_train_sample(min(360000, sample_size_limit))
            input_data = self.training_sampler.obtain_input_data(topics_id=topics, contents_id=contents)
            y = tf.constant(cors)
            y_pred = model(input_data)
            for j in range(4):
                kmetrics[j].update_state(y, y_pred)
    def obtain_metrics(self):
        metrics_list = [metr for met in self.metrics for metr in met["metrics"]]
        return {m.name: m.result() for m in metrics_list}

    def get_test_entropy_metric(self):
        for k in range(len(self.metrics)):
            mmetrics = self.metrics[k]["metrics"]
            if mmetrics[3].name == "test_entropy":
                return mmetrics[3].result()
        raise Exception("No metrics found!")

default_metrics = DynamicMetrics()
default_metrics.add_metric("test", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST)
default_metrics.add_metric("test_square", data_bert_sampler.default_sampler_instance, sample_choice = DynamicMetrics.TEST_SQUARE)

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