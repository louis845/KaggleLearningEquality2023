import numpy as np
import tensorflow as tf
import time as time
import gc
import model_bert_evaluation_pipeline
def get_topk_gpu(x, k):
    topk_preds = tf.math.top_k(x, k=k, sorted=True)
    return tf.reverse(topk_preds.indices, [-1])

# BASICALLY, run from top to bottom
# algorithm for dot product topk
def obtain_rowwise_topk_from_dot_prod(topics_restrict, contents_restrict, full_topics_data, full_contents_data, topk = 100,
                                      greedy_multiple_rows = 10000, max_batch_size = 100000):
    full_contents_restricted_matrix = tf.gather(full_contents_data, contents_restrict, axis=0)
    length = len(topics_restrict)

    print("Allocating topk_preds....")
    topk_preds = np.zeros(shape=(len(topics_restrict), topk), dtype=np.int32)
    print("Allocated topk_preds.")

    batch_size = greedy_multiple_rows
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    print("Running rowwise topk loop....")
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_id_rows = topics_restrict[np.arange(tlow, thigh)]
        try:
            dotsim = tf.linalg.matmul(tf.gather(full_topics_data, topic_id_rows, axis=0), full_contents_restricted_matrix, transpose_b=True)
            topk_ascending = get_topk_gpu(dotsim, topk)
        except tf.errors.ResourceExhaustedError as err:
            topk_ascending = None
        if topk_ascending is not None:
            topk_ascending_np = topk_ascending.numpy()
            del topk_ascending, dotsim
            topk_preds[np.arange(tlow, thigh), :] = contents_restrict[topk_ascending_np]
            del topk_ascending_np
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 2000, max_batch_size)

            if tlow - prev_tlow > 4000:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 2000, 1)
            max_batch_size = batch_size
            continuous_success = 0
        gc.collect()

    return topk_preds

def restrict_tuple_with_simscore(topics_tuple, contents_tuple, full_topics_data, full_contents_data, threshold=0.1,
                                         batch_size=1000000):
    simscore = model_bert_evaluation_pipeline.obtain_dot_prod_simscore_from_tuples(topics_tuple, contents_tuple, full_topics_data, full_contents_data,
                                         batch_size=batch_size)
    gc.collect()
    places = simscore > threshold
    topics_tuple2, contents_tuple2 = topics_tuple[places], contents_tuple[places]
    del places
    gc.collect()
    return topics_tuple2, contents_tuple2

def obtain_tuple_based_probas_stepup_dimreduce(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, batch_size=400000):
    max_batch_size = np.inf
    length = len(topics_tuple)
    assert length == len(contents_tuple)
    print_length = max(length // 50, 1)

    total_probabilities = np.zeros(shape=length, dtype=np.float32)
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_ids = topics_tuple[np.arange(tlow, thigh)]
        content_ids = contents_tuple[np.arange(tlow, thigh)]
        try:
            probabilities = model_bert_evaluation_pipeline.predict_probabilities_direct_gpu_stepup_dimreduce(proba_callback, tf.constant(topic_ids), tf.constant(content_ids),
                                                             full_topics_d1, full_contents_d1,
                                                             full_topics_d1fp, full_contents_d1fp)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            probabilities_np = probabilities.numpy()
            total_probabilities[tlow:thigh] = probabilities_np
            del probabilities, probabilities_np
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 50000, max_batch_size)

            if tlow - prev_tlow > print_length:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 50000, 1)
            max_batch_size = batch_size
            continuous_success = 0
        del topic_ids, content_ids
        gc.collect()

    assert len(total_probabilities) == length
    return total_probabilities
def obtain_tuple_based_probas_stepup_dimreduce2(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, batch_size=400000):
    max_batch_size = np.inf
    length = len(topics_tuple)
    assert length == len(contents_tuple)
    print_length = max(length // 50, 1)

    total_probabilities = np.zeros(shape=length, dtype=np.float32)
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = time.time()
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_ids = topics_tuple[np.arange(tlow, thigh)]
        content_ids = contents_tuple[np.arange(tlow, thigh)]
        try:
            probabilities = model_bert_evaluation_pipeline.predict_probabilities_direct_gpu_stepup_dimreduce2(proba_callback, tf.constant(topic_ids), tf.constant(content_ids),
                                                             full_topics_d1, full_contents_d1,
                                                             full_topics_d1fp, full_contents_d1fp)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            probabilities_np = probabilities.numpy()
            total_probabilities[tlow:thigh] = probabilities_np
            del probabilities, probabilities_np
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 50000, max_batch_size)

            if tlow - prev_tlow > print_length:
                ctime = time.time() - ctime
                print(tlow, "completed. out of:", length, "  batch size:", batch_size, "  time used:", ctime)
                prev_tlow = tlow
                ctime = time.time()
        else:
            batch_size = max(batch_size - 50000, 1)
            max_batch_size = batch_size
            continuous_success = 0
        del topic_ids, content_ids
        gc.collect()

    assert len(total_probabilities) == length
    return total_probabilities

def restrict_tuple_with_probas(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, threshold=0.5, batch_size=70000):
    probas = obtain_tuple_based_probas_stepup_dimreduce(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, batch_size=batch_size)
    gc.collect()
    places = probas > threshold
    topics_tuple2, contents_tuple2 = topics_tuple[places], contents_tuple[places]
    del places
    gc.collect()
    return topics_tuple2, contents_tuple2

def restrict_tuple_with_probas2(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, threshold=0.5, batch_size=70000):
    probas = obtain_tuple_based_probas_stepup_dimreduce2(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, batch_size=batch_size)
    gc.collect()
    places = probas > threshold
    topics_tuple2, contents_tuple2 = topics_tuple[places], contents_tuple[places]
    del places
    gc.collect()
    return topics_tuple2, contents_tuple2