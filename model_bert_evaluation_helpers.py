import data_bert
import numpy as np
import time
import pandas as pd

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
            row_preds = set(list(topk_pred[topics_restrict_id]))

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