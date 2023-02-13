"""
A model for classifying topic-content correlations.

Model input: dimensions (batch_size x set_size x (bert_embedding_size*2+num_langs+bert_embedding_size*2+num_langs)) (ragged tensor).

The 0th axis is the batch size. The 1st axis is the set_size for the current input, which may vary for each different instance in the
batch. The 2nd axis are the precomputed inputs from BERT embedding concatenated with the one-hot representation of languages. The order
is (content_description, content_title, content_lang, topic_description, topic_title, topic_lang).

Each input [k,:,:] denotes the kth input to model, which is a set of topics and contents tuples. The content must be the same throughout
the set [k,:,:]. Note [k,j,:] denotes the tuple (content_description, content_title, content_lang, topic_description, topic_title, topic_lang)
belonging to the jth element in the set of kth input to the model.

Model output:
A (batch_size) tensor (vector) containing the predicted probabilities. The model tries to predict whether the set of topics contain the
given content.
"""


import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, units_size=512):
        super(Model, self).__init__()

        # concatenation layer
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)

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

    def call(self, inputs, training=False):
        content_description, content_title, topic_description, topic_title = inputs

        embedding_result = self.concat_layer([content_description, content_title, topic_description, topic_title])
        t = self.dropout1(self.dense1(embedding_result), training=training)
        t = self.dropout2(self.dense2(t), training=training)
        t = self.dropout3(self.dense3(t), training=training)
        return self.dense4(t)

    def train_step(self, data):
        for k in range(50):
            topics, contents, cors = data_bert.obtain_train_sample(self.training_one_sample_size,
                                                                   self.training_zero_sample_size)
            y = tf.constant(cors)
            content_description, content_title, topic_description, topic_title = obtain_embedded_vects(contents, topics)

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