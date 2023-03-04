# this is not the full pipeline. given the vector information (either from BERT or my custom models, output the predicted
# contents and topics)

import numpy as np
import config
import pandas as pd
import time
import data_bert
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

def evaluate_topk_from_performance(topk_preds, topics_restrict, contents_restrict):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_preds.keys():
            topk_pred = topk_preds[k]
            row_preds = set(topk_pred[topics_restrict_id, :])

            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()]))
                actual_cors = set(actual_cors[data_bert.fast_contains_multi(contents_restrict, actual_cors)])
                row_tp = len(row_preds.intersection(actual_cors))
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0:
                row_recall, row_precision, row_f2 = 0,0,0
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
                row_f2 = 5.0 * row_precision * row_recall / (4.0 * row_precision + row_recall)
            row_accuracy = (row_tp + row_tn + 0.0) / len(contents_restrict)

            true_positive[k] += row_tp
            true_negative[k] += row_tn
            false_positive[k] += row_fp
            false_negative[k] += row_fn
            total_recall[k] += row_recall
            total_precision[k] += row_precision
            total_accuracy[k] += row_accuracy
            total_f2[k] += row_f2

        if topics_restrict_id % 200 == 0:
            print("Computed metrics of " + str(topics_restrict_id) + " out of " + str(len(topics_restrict)))
            ctime = time.time() - ctime
            print(ctime)
            ctime = time.time()

    info_index = list(topk_preds.keys())
    metrics_array = np.zeros(shape = (len(info_index), 8))
    for kidx in range(len(info_index)):
        k = info_index[kidx]

        if true_positive[k] == 0:
            final_recall, final_precision, final_f2 = 0, 0, 0
        else:
            final_recall = (true_positive[k] + 0.0) / (true_positive[k] + false_negative[k])
            final_precision = (true_positive[k] + 0.0) / (true_positive[k] + false_positive[k])
            final_f2 = 5.0 * final_precision * final_recall / (4.0 * final_precision + final_recall)
        final_accuracy = (true_positive[k] + true_negative[k] + 0.0) / (len(contents_restrict) * len(topics_restrict))

        mean_recall = total_recall[k] / len(topics_restrict)
        mean_precision = total_precision[k] / len(topics_restrict)
        mean_accuracy = total_accuracy[k] / len(topics_restrict)
        mean_f2 = total_f2[k] / len(topics_restrict)

        metrics_array[kidx, 0] = mean_recall
        metrics_array[kidx, 1] = mean_precision
        metrics_array[kidx, 2] = mean_accuracy
        metrics_array[kidx, 3] = mean_f2
        metrics_array[kidx, 4] = final_recall
        metrics_array[kidx, 5] = final_precision
        metrics_array[kidx, 6] = final_f2
        metrics_array[kidx, 7] = final_accuracy
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy"])