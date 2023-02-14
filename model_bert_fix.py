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

class TrainingSampler:
    # loads the data as tf tensors from the folders specified
    def __init__(self, embedded_vectors_folder, contents_one_hot_file, topics_one_hot_file):
        self.contents_description = tf.constant(np.load(embedded_vectors_folder + "contents_description.npy"))
        self.contents_title = tf.constant(np.load(embedded_vectors_folder + "contents_title.npy"))
        self.topics_description = tf.constant(np.load(embedded_vectors_folder + "topics_description.npy"))
        self.topics_title = tf.constant(np.load(embedded_vectors_folder + "topics_title.npy"))

        self.contents_one_hot = tf.constant(np.load(contents_one_hot_file))
        self.topics_one_hot = tf.constant(np.load(topics_one_hot_file))


    # obtains batch_size x 1 x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs) tensor, along with np vector
    # of size batch_size. return format is {"contents":{"description":, "title":, "lang":}, "topics":{"description":, "title":, "lang":}}, actual_y
    # samples zero_sample_size zeros, one_sample_size ones from the training data.
    def obtain_train_data(self, zero_sample_size, one_sample_size):
        topics, contents, cors = data_bert.obtain_train_sample(one_sample_size = one_sample_size, zero_sample_size = zero_sample_size)
        input_data = {
         "contents":{
             "description":tf.gather(self.contents_description, contents, axis = 0),
             "title":tf.gather(self.contents_title, contents, axis = 0),
             "lang":tf.gather(self.contents_one_hot, contents, axis = 0)
         },
         "topics":{
             "description":tf.gather(self.topics_description, topics, axis = 0),
             "title":tf.gather(self.topics_title, topics, axis = 0),
             "lang":tf.gather(self.topics_one_hot, topics, axis = 0)}
        }
        return input_data, cors

    # obtains batch_size x 1 x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs) tensor, along with np vector
    # of size batch_size. return format is {"contents":{"description":, "title":, "lang":}, "topics":{"description":, "title":, "lang":}}, actual_y
    # samples a sample_size x sample_size region of topics and contents from the training data.
    def obtain_train_data_full(self):
        return None

    # obtains batch_size x 1 x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs) tensor, along with np vector
    # of size batch_size. return format is {"contents":{"description":, "title":, "lang":}, "topics":{"description":, "title":, "lang":}}, actual_y
    # samples zero_sample_size zeros, one_sample_size ones from the test data.
    def obtain_test_data(self, zero_sample_size, one_sample_size):
        return None

    # obtains batch_size x 1 x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs) tensor, along with np vector
    # of size batch_size. return format is {"contents":{"description":, "title":, "lang":}, "topics":{"description":, "title":, "lang":}}, actual_y
    # samples a sample_size x sample_size region of topics and contents from the test data.
    def obtain_test_data_full(self):
        return None

class Model(tf.keras.Model):
    # only the argument units_size define the shape of the model. the argument training_sampler is used for training only.
    def __init__(self, units_size = 512, training_sampler = None):
        super(Model, self).__init__()

        self.training_sampler = training_sampler

        # concatenation layer (acts on the final axis, meaning putting together contents_description, content_title, content_lang etc..)
        self.concat_layer = tf.keras.layers.Concatenate(axis = 2)

        # standard stuff
        self.dense1 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.05)
        self.dense2 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(rate=0.05)
        self.dense3 = tf.keras.layers.Dense(units=units_size, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(rate=0.05)
        self.dense4 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        # loss functions and eval metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision = tf.keras.metrics.Precision(name="precision")
        self.recall = tf.keras.metrics.Recall(name="recall")
        self.entropy = tf.keras.metrics.BinaryCrossentropy(name="entropy")
        self.entropy_large_set = tf.keras.metrics.BinaryCrossentropy(name="entropy_large_set")

        # metrics for test set
        threshold = 0.5
        self.full_accuracy = tf.keras.metrics.BinaryAccuracy(name="full_accuracy", threshold=threshold)
        self.full_precision = tf.keras.metrics.Precision(name="full_precision", thresholds=threshold)
        self.full_recall = tf.keras.metrics.Recall(name="full_recall", thresholds=threshold)
        self.full_entropy = tf.keras.metrics.BinaryCrossentropy(name="full_entropy")

        self.test_precision = tf.keras.metrics.Precision(name="test_precision", thresholds=threshold)
        self.test_recall = tf.keras.metrics.Recall(name="test_recall", thresholds=threshold)

        self.test_small_precision = tf.keras.metrics.Precision(name="test_small_precision", thresholds=threshold)
        self.test_small_recall = tf.keras.metrics.Recall(name="test_small_recall", thresholds=threshold)
        self.test_small_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_small_accuracy", threshold=threshold)
        self.test_small_entropy = tf.keras.metrics.BinaryCrossentropy(name="test_small_entropy")

    def compile(self):
        super(Model, self).compile(run_eagerly=True)
        # loss and optimizer
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.training_one_sample_size = 1000
        self.training_zero_sample_size = 1000

    def set_training_params(self, training_zero_sample_size=1000, training_one_sample_size=1000):
        self.training_one_sample_size = training_one_sample_size
        self.training_zero_sample_size = training_zero_sample_size

    # for training, we feed in actual_y to overdetermine the predictions. if actual_y is not fed in,
    # usual gradient descent will be used. actual_y should be a (batch_size) numpy vector.
    def call(self, data, training=False, actual_y = None):
        contents_description = data["contents"]["description"]
        contents_title = data["contents"]["title"]
        contents_lang = data["contents"]["lang"]
        topics_description = data["topics"]["description"]
        topics_title = data["topics"]["title"]
        topics_lang = data["topics"]["lang"]
        # combine the (batch_size x set_size x (bert_embedding_size / num_langs)) tensors into (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs))
        embedding_result = self.concat_layer([contents_description, contents_title, contents_lang, topics_description, topics_title, topics_lang])

        t = self.dropout1(self.dense1(embedding_result), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        t = self.dense4(t) # now we have a batch_size x set_size x 1 tensor, the last axis is reduced to 1 by linear transforms.
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
            proba = tf.math.multiply(pinvmean * tf.constant(actual_y), pmean * tf.constant(1 - actual_y))
            return proba
        else: # here we just return the probabilities normally. the probability will be computed as the max inside the set
            return tf.squeeze(tf.reduce_max(t, axis = 1), axis = 1)


    def train_step(self, data):
        for k in range(50):
            input_data, actual_y = training_sampler.obtain_train_data()

            with tf.GradientTape() as tape:
                y_pred = self((content_description, content_title, topic_description, topic_title), training=True)
                loss = self.loss(y, y_pred)

            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for m in self.metrics:
            m.update_state(y, y_pred)

        # evaluation at larger subset
        topics, contents, cors = data_bert.obtain_train_sample(len(data_bert.train_contents),
                                                               len(data_bert.train_contents))
        y = tf.constant(cors)
        content_description, content_title, topic_description, topic_title = obtain_embedded_vects(contents, topics)

        y_pred = self((content_description, content_title, topic_description, topic_title))
        self.entropy_large_set.update_state(y, y_pred)

        # evaluation at other points
        topics, contents, cors = data_bert.obtain_train_square_sample(600)
        y = tf.constant(cors)
        content_description, content_title, topic_description, topic_title = obtain_embedded_vects(contents, topics)

        y_pred = self((content_description, content_title, topic_description, topic_title))

        for m in self.full_metrics:
            m.update_state(y, y_pred)

        topics, contents, cors = data_bert.obtain_test_square_sample(600)
        y = tf.constant(cors)
        content_description, content_title, topic_description, topic_title = obtain_embedded_vects(contents, topics)

        y_pred = self((content_description, content_title, topic_description, topic_title))

        for m in self.test_metrics:
            m.update_state(y, y_pred)

        topics, contents, cors = data_bert.obtain_test_sample(30000, 30000)
        y = tf.constant(cors)
        content_description, content_title, topic_description, topic_title = obtain_embedded_vects(contents, topics)

        y_pred = self((content_description, content_title, topic_description, topic_title))

        for m in self.test_small_metrics:
            m.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics}, **{m.name: m.result() for m in self.full_metrics},
                **{m.name: m.result() for m in self.test_metrics},
                **{m.name: m.result() for m in self.test_small_metrics},
                "entropy_large_set": entropy_large_set.result()}

    @property
    def metrics(self):
        return [self.accuracy, self.precision, self.recall, self.entropy]

    @property
    def test_metrics(self):
        return [self.test_precision, self.test_recall]

    @property
    def test_small_metrics(self):
        return [self.test_small_accuracy, self.test_small_precision, self.test_small_recall, self.test_small_entropy]

    @property
    def full_metrics(self):
        return [self.full_accuracy, self.full_precision, self.full_recall, self.full_entropy]