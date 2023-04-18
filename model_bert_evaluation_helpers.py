import data_bert
import numpy as np
import time
import pandas as pd
import os

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

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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
def evaluate_topk_from_performance_format2(topk_preds, topics_restrict, contents_restrict):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_preds.keys():
            topk_pred = topk_preds[k]
            row_preds = np.unique(topk_pred[topics_restrict_id])

            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.unique(np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()])))
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

def evaluate_topk_from_perfomance_tuple(topics_tuple, contents_tuple, topics_restrict, contents_restrict):
    assert (topics_tuple[1:] < topics_tuple[:-1]).sum() == 0
    assert len(topics_tuple) == len(contents_tuple)

    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = 0,0,0,0
    total_recall, total_precision, total_accuracy, total_f2 = 0,0,0,0

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        left = np.searchsorted(topics_tuple, topic_id, side="left")
        right = np.searchsorted(topics_tuple, topic_id, side="right")
        if left == right:
            row_preds = np.array([], dtype=np.int32)
        else:
            contents_tuple_res = contents_tuple[left:right]
            row_preds = np.unique(contents_tuple_res)

        row_tp, row_tn, row_fp, row_fn = 0,0,0,0
        if data_bert.topics.loc[topic_str_id, "has_content"]:
            actual_cors = np.unique(np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()])))
            row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
            row_fn = len(actual_cors) - row_tp
            row_fp = len(row_preds) - row_tp
            row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
        else:
            row_fp = len(row_preds)
            row_tn = len(contents_restrict) - row_fp
            # here true positive and false negative are zero

        if row_tp == 0 and row_fn == 0:
            row_recall = 1
        else:
            row_recall = (row_tp + 0.0) / (row_tp + row_fn)
        if row_tp == 0 and row_fp == 0:
            if row_tp == 0 and row_fn == 0:
                row_precision = 1
            else:
                row_precision = 0
        else:
            row_precision = (row_tp + 0.0) / (row_tp + row_fp)
        if row_precision == 0 and row_recall == 0:
            row_f2 = 0
        else:
            row_f2 = 5.0 * row_precision * row_recall / (4.0 * row_precision + row_recall)
        row_accuracy = (row_tp + row_tn + 0.0) / len(contents_restrict)

        true_positive += row_tp
        true_negative += row_tn
        false_positive += row_fp
        false_negative += row_fn
        total_recall += row_recall
        total_precision += row_precision
        total_accuracy += row_accuracy
        total_f2 += row_f2

        if topics_restrict_id % 200 == 0:
            print("Computed metrics of " + str(topics_restrict_id) + " out of " + str(len(topics_restrict)))
            ctime = time.time() - ctime
            print(ctime)
            ctime = time.time()

    info_index = ["Tuple:"]
    metrics_array = np.zeros(shape = (len(info_index), 8))
    for kidx in range(len(info_index)):
        if true_positive == 0:
            final_recall, final_precision, final_f2 = 0, 0, 0
        else:
            final_recall = (true_positive + 0.0) / (true_positive + false_negative)
            final_precision = (true_positive + 0.0) / (true_positive + false_positive)
            final_f2 = 5.0 * final_precision * final_recall / (4.0 * final_precision + final_recall)
        final_accuracy = (true_positive + true_negative + 0.0) / (len(contents_restrict) * len(topics_restrict))

        mean_recall = total_recall / len(topics_restrict)
        mean_precision = total_precision / len(topics_restrict)
        mean_accuracy = total_accuracy / len(topics_restrict)
        mean_f2 = total_f2 / len(topics_restrict)

        metrics_array[kidx, 0] = mean_recall
        metrics_array[kidx, 1] = mean_precision
        metrics_array[kidx, 2] = mean_accuracy
        metrics_array[kidx, 3] = mean_f2
        metrics_array[kidx, 4] = final_recall
        metrics_array[kidx, 5] = final_precision
        metrics_array[kidx, 6] = final_f2
        metrics_array[kidx, 7] = final_accuracy
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy"])

def evaluate_topks_from_perfomance_tuple(topics_tuple, contents_tuple, topics_restrict, contents_restrict, topks = (np.arange(50) + 1)*2):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topks, 0), dict.fromkeys(topks, 0), dict.fromkeys(topks, 0), dict.fromkeys(topks, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topks, 0), dict.fromkeys(topks, 0), dict.fromkeys(topks, 0), dict.fromkeys(topks, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        left = np.searchsorted(topics_tuple, topic_id, side="left")
        right = np.searchsorted(topics_tuple, topic_id, side="right")
        if left == right:
            row_preds_full = np.array([], dtype=np.int32)
        else:
            contents_tuple_res = contents_tuple[left:right]
            row_preds_full = contents_tuple_res

        for k in topks:
            if k < len(row_preds_full):
                row_preds = np.unique(row_preds_full[-k:])
            else:
                row_preds = np.unique(row_preds_full)

            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.unique(np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()])))
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

    info_index = list(topks)
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

def evaluate_topk_from_performance_with_dotsim_restrict(topk_preds, dotsim_preds, topics_restrict, contents_restrict):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0), dict.fromkeys(topk_preds, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_preds.keys():
            topk_pred = topk_preds[k]
            row_preds = topk_pred[topics_restrict_id, :]
            dotsim_predslc = dotsim_preds[k][topics_restrict_id, :]
            row_preds = np.unique(row_preds[dotsim_predslc > 0])

            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.unique(np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()])))
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

# topics_overshoot_mapping[k] is a mapping from topic_num_id to another topic_num_id, (node to another node), where the other
# node is the node's ancestor (or itself) such that the subtree has a certain size.
# topics_overshoot_mapping is a dict with integer keys, the key is the min size.
def evaluate_quality_from_cors_arr(topics_overshoot_mapping, topics_restrict, contents_restrict, cors_arr):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0), dict.fromkeys(topics_overshoot_mapping, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topics_overshoot_mapping.keys():
            mapping_topic_id = topics_overshoot_mapping[k][topic_id]
            row_preds = cors_arr[mapping_topic_id]

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

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

    info_index = list(topics_overshoot_mapping.keys())
    metrics_array = np.zeros(shape = (len(info_index), 9))
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
        metrics_array[kidx, 8] = (true_positive[k] + false_positive[k] + 0.0) / (true_positive[k] + false_positive[k] + true_negative[k] + false_negative[k])
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy", "restriction_ratio"])

def evaluate_diff_topk_intersection_quality(topics_restrict, contents_restrict, cors_arr, topk_restriction_matrix, topk_vals):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_vals:
            row_preds = np.unique(cors_arr[topic_id])
            row_preds2 = topk_restriction_matrix[topics_restrict_id, -k:]
            row_preds = np.unique(row_preds2[data_bert.fast_contains_multi(row_preds, row_preds2)])


            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()]))
                actual_cors = actual_cors[data_bert.fast_contains_multi(contents_restrict, actual_cors)]
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

    info_index = list(topk_vals)
    metrics_array = np.zeros(shape = (len(info_index), 9))
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
        metrics_array[kidx, 8] = (true_positive[k] + false_positive[k] + 0.0) / (true_positive[k] + false_positive[k] + true_negative[k] + false_negative[k])
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy", "restriction_ratio"])

def evaluate_diff_topk_intersection_quality(topics_restrict, contents_restrict, cors_arr, topk_restriction_matrix, topk_vals):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_vals:
            row_preds = np.unique(cors_arr[topic_id])
            row_preds2 = topk_restriction_matrix[topics_restrict_id, -k:]
            row_preds = np.unique(row_preds2[data_bert.fast_contains_multi(row_preds, row_preds2)])


            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()]))
                actual_cors = actual_cors[data_bert.fast_contains_multi(contents_restrict, actual_cors)]
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

    info_index = list(topk_vals)
    metrics_array = np.zeros(shape = (len(info_index), 9))
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
        metrics_array[kidx, 8] = (true_positive[k] + false_positive[k] + 0.0) / (true_positive[k] + false_positive[k] + true_negative[k] + false_negative[k])
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy", "restriction_ratio"])

def evaluate_diff_topk_intersection_quality_restrict(topics_restrict, contents_restrict, cors_arr, topk_restriction_matrix, topk_vals):
    contents_restrict = np.sort(contents_restrict)
    true_positive, true_negative, false_positive, false_negative = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)
    total_recall, total_precision, total_accuracy, total_f2 = dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0), dict.fromkeys(topk_vals, 0)

    ctime = time.time()
    for topics_restrict_id in range(len(topics_restrict)):
        topic_id = topics_restrict[topics_restrict_id]
        topic_str_id = data_bert.topics.index[topic_id]

        for k in topk_vals:
            row_preds = np.unique(cors_arr[topic_id])
            row_preds2 = topk_restriction_matrix[topics_restrict_id, :]
            row_preds = row_preds2[data_bert.fast_contains_multi(row_preds, row_preds2)]
            if k < len(row_preds):
                row_preds = row_preds[-k:]

            row_preds = np.unique(row_preds)


            row_tp, row_tn, row_fp, row_fn = 0,0,0,0
            if data_bert.topics.loc[topic_str_id, "has_content"]:
                actual_cors = np.array(list(data_bert.contents_inv_map.loc[data_bert.correlations.loc[topic_str_id, "content_ids"].split()]))
                actual_cors = actual_cors[data_bert.fast_contains_multi(contents_restrict, actual_cors)]
                row_tp = data_bert.fast_contains_multi(row_preds, actual_cors).sum()
                row_fn = len(actual_cors) - row_tp
                row_fp = len(row_preds) - row_tp
                row_tn = len(contents_restrict) - row_tp - row_fn - row_fp
            else:
                row_fp = len(row_preds)
                row_tn = len(contents_restrict) - row_fp
                # here true positive and false negative are zero

            if row_tp == 0 and row_fn == 0:
                row_recall = 1
            else:
                row_recall = (row_tp + 0.0) / (row_tp + row_fn)
            if row_tp == 0 and row_fp == 0:
                if row_tp == 0 and row_fn == 0:
                    row_precision = 1
                else:
                    row_precision = 0
            else:
                row_precision = (row_tp + 0.0) / (row_tp + row_fp)
            if row_precision == 0 and row_recall == 0:
                row_f2 = 0
            else:
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

    info_index = list(topk_vals)
    metrics_array = np.zeros(shape = (len(info_index), 9))
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
        metrics_array[kidx, 8] = (true_positive[k] + false_positive[k] + 0.0) / (true_positive[k] + false_positive[k] + true_negative[k] + false_negative[k])
    return pd.DataFrame(data = metrics_array, index = pd.Index(info_index), columns = ["mean_recall", "mean_precision", "mean_accuracy", "mean_f2", "final_recall", "final_precision", "final_f2", "final_accuracy", "restriction_ratio"])