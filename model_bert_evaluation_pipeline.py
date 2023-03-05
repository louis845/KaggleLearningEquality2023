# this is not the full pipeline. given the vector information (either from BERT or my custom models, output the predicted
# contents and topics)

import numpy as np
import time
import math
import gc
import tensorflow as tf

class ObtainProbabilitiesCallback:
    def __init__(self):
        pass

    # topics_vector, contents_vector are n x k arrays, where their k can be different (depending on model)
    # the first axis (n) is the batch_size axis, which means the prediction function predicts n probabilities
    # the second axis (k) is the vector embedding axis, which is either obtained from BERT or other models
    # if succeeded, return the probabilities. if failed, return None.
    def predict_probabilities(self, unified_topics_contents_vector, device):
        pass

    # same thing as above, except that we return the GPU information directly (and of course use GPU for computation).
    def predict_probabilities_return_gpu(self, unified_topics_contents_vector):
        pass

    # helper function, do not override.
    def predict_probabilities_with_data(self, topics_id, contents_id, full_topics_vect_data, full_contents_vect_data, device):
        if device == "gpu":
            topics_vector = tf.gather(full_topics_vect_data, topics_id, axis = 0)
            contents_vector = tf.gather(full_contents_vect_data, contents_id, axis = 0)
            return self.predict_probabilities(tf.concat([contents_vector, topics_vector], axis=1), device)
        else:
            topics_vector = full_topics_vect_data[topics_id,:]
            contents_vector = full_contents_vect_data[contents_id,:]
            return self.predict_probabilities(np.concatenate([contents_vector, topics_vector], axis = 1), device)

    # helper function, do not override.
    def predict_probabilities_with_data_return_gpu(self, topics_id, contents_id, full_topics_vect_data,
                                        full_contents_vect_data):
        topics_vector = tf.gather(full_topics_vect_data, topics_id, axis=0)
        contents_vector = tf.gather(full_contents_vect_data, contents_id, axis=0)
        return self.predict_probabilities_return_gpu(tf.concat([contents_vector, topics_vector], axis=1))

def predict_rows(proba_callback, topic_id_rows, contents_restrict, full_topics_data, full_contents_data, device):
    if device == "gpu":
        topics_id = tf.repeat(tf.constant(topic_id_rows), len(contents_restrict))
        contents_id = tf.tile(tf.constant(contents_restrict), [len(topic_id_rows)])
    else:
        topics_id = np.repeat(topic_id_rows, len(contents_restrict))
        contents_id = np.tile(contents_restrict, len(topic_id_rows))
    probabilities = proba_callback.predict_probabilities_with_data(topics_id, contents_id, full_topics_data,
                                                                   full_contents_data, device)
    return probabilities

@tf.function
def predict_rows_gpu(proba_callback, topic_id_rows, contents_restrict, full_topics_data, full_contents_data):
    topics_id = tf.repeat(topic_id_rows, contents_restrict.shape[0])
    contents_id = tf.tile(contents_restrict, [topic_id_rows.shape[0]])
    probabilities = proba_callback.predict_probabilities_with_data_return_gpu(topics_id, contents_id, full_topics_data,
                                                                   full_contents_data)
    return probabilities

default_topk_values = (np.arange(40) + 1) * 3  # TODO - find optimal topk

def get_topk(x, k):
    res = np.argpartition(x, kth = -k, axis = 1)[:, -k:]
    rep = np.repeat(np.expand_dims(np.arange(res.shape[0]), axis = 1), res.shape[1], axis = 1)
    res2 = np.argsort(x[rep, res], axis = 1)
    return res[rep, res2]

# topics_restrict, contents_restrict are np arrays containing the restrictions to topics and contents respectively
# usually this is used to restrict it to test set. topk_values are the topk probas for the model to choose from.
# by default, it is
def obtain_rowwise_topk(proba_callback, topics_restrict, contents_restrict, full_topics_data, full_contents_data, topk_values = None, greedy_multiple_rows = 40, device = "gpu", max_batch_size = 40):
    if topk_values is None:
        topk_values = default_topk_values

    # dict of np arrays, where each np array is len(topics_restrict) x topk_values[i], where each row contains the topk predictions
    topk_preds = {}
    for i in range(len(topk_values)):
        topk_preds[topk_values[i]] = np.zeros(shape = (len(topics_restrict), topk_values[i]))

    length = len(topics_restrict)
    max_topk = np.max(topk_values)

    batch_size = greedy_multiple_rows
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_id_rows = topics_restrict[np.arange(tlow, thigh)]
        try:
            probabilities = predict_rows(proba_callback, topic_id_rows, contents_restrict, full_topics_data,
                                         full_contents_data, device = device)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            probabilities = probabilities.reshape((thigh - tlow), len(contents_restrict))
            sorted_locs = get_topk(probabilities, max_topk)
            for i in range(len(topk_values)):
                topk_preds[topk_values[i]][np.arange(tlow, thigh), :] = contents_restrict[
                    sorted_locs[:, -topk_values[i]:]]
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 1, max_batch_size)

            if tlow - prev_tlow > 50:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 1, 1)
            max_batch_size = batch_size
            continuous_success = 0
        gc.collect()

    return topk_preds

# topics_restrict, contents_restrict are np arrays containing the restrictions to topics and contents respectively
# usually this is used to restrict it to test set. topk_values are the topk probas for the model to choose from.
# by default, it is
def obtain_rowwise_topk_pgpu(proba_callback, topics_restrict, contents_restrict, full_topics_data, full_contents_data, topk_values = None, greedy_multiple_rows = 40, max_batch_size = 40):
    if topk_values is None:
        topk_values = default_topk_values

    # dict of np arrays, where each np array is len(topics_restrict) x topk_values[i], where each row contains the topk predictions
    topk_preds = {}
    for i in range(len(topk_values)):
        topk_preds[topk_values[i]] = np.zeros(shape = (len(topics_restrict), topk_values[i]))

    length = len(topics_restrict)
    max_topk = np.max(topk_values)

    batch_size = greedy_multiple_rows
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_id_rows = topics_restrict[np.arange(tlow, thigh)]
        try:
            probabilities = predict_rows_gpu(proba_callback, tf.constant(topic_id_rows), tf.constant(contents_restrict), full_topics_data,
                                         full_contents_data)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            probabilities = probabilities.numpy().reshape((thigh - tlow), len(contents_restrict))
            sorted_locs = get_topk(probabilities, max_topk)
            for i in range(len(topk_values)):
                topk_preds[topk_values[i]][np.arange(tlow, thigh), :] = contents_restrict[
                    sorted_locs[:, -topk_values[i]:]]
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 1, max_batch_size)

            if tlow - prev_tlow > 50:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 1, 1)
            max_batch_size = batch_size
            continuous_success = 0
        gc.collect()

    return topk_preds

@tf.function
def predict_probabilities_direct_gpu(proba_callback, topics_tuple, contents_tuple, full_topics_data, full_contents_data):
    return proba_callback.predict_probabilities_with_data_return_gpu(topics_tuple, contents_tuple, full_topics_data, full_contents_data)

def obtain_tuple_based_probas(proba_callback, topics_tuple, contents_tuple, full_topics_data, full_contents_data,
                              batch_size=70000):
    max_batch_size = np.inf
    length = len(topics_tuple)
    assert length == len(contents_tuple)
    print_length = max(length // 50, 1)

    total_probabilities = np.zeros(shape=0, dtype=np.float32)
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_ids = topics_tuple[np.arange(tlow, thigh)]
        content_ids = contents_tuple[np.arange(tlow, thigh)]
        try:
            probabilities = predict_probabilities_direct_gpu(proba_callback, tf.constant(topic_ids), tf.constant(content_ids),
                                             full_topics_data,
                                             full_contents_data)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            pprob = total_probabilities
            total_probabilities = np.concatenate([pprob, probabilities.numpy()])
            del pprob, probabilities
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 30000, max_batch_size)

            if tlow - prev_tlow > print_length:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 1500, 1)
            max_batch_size = batch_size
            continuous_success = 0
        gc.collect()

    assert len(total_probabilities) == length
    return total_probabilities