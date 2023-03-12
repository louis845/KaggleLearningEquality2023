# this is not the full pipeline. given the vector information (either from BERT or my custom models, output the predicted
# contents and topics)
import os.path

import numpy as np
import time
import math
import gc
import tensorflow as tf

class ObtainProbabilitiesCallback:
    def __init__(self):
        self.model = None

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

    def predict_probabilities_with_data_return_gpu_dimreduce(self, topics_id, contents_id, full_topics_d1,
                                                             full_contents_d1, full_topics_d1fp, full_contents_d1fp):
        topics_d1 = tf.gather(full_topics_d1, topics_id, axis=0)
        contents_d1 = tf.gather(full_contents_d1, contents_id, axis=0)
        topics_d1fp = tf.gather(full_topics_d1fp, topics_id, axis=0)
        contents_d1fp = tf.gather(full_contents_d1fp, contents_id, axis=0)
        return self.model.call_fast_dim_reduced(contents_d1, topics_d1, contents_d1fp, topics_d1fp)

    def predict_probabilities_with_data_return_gpu_3dimreduce(self, topics_id, contents_id, full_topics_d1os,
                                                             full_contents_d1os, full_topics_d1dp, full_contents_d1dp,
                                                              full_topics_d1fp, full_contents_d1fp):
        topics_d1os = tf.gather(full_topics_d1os, topics_id, axis=0)
        contents_d1os = tf.gather(full_contents_d1os, contents_id, axis=0)
        topics_d1dp = tf.gather(full_topics_d1dp, topics_id, axis=0)
        contents_d1dp = tf.gather(full_contents_d1dp, contents_id, axis=0)
        topics_d1fp = tf.gather(full_topics_d1fp, topics_id, axis=0)
        contents_d1fp = tf.gather(full_contents_d1fp, contents_id, axis=0)
        return self.model.call_fast_dim_reduced(contents_d1os, topics_d1os, contents_d1dp,
                                                topics_d1dp, contents_d1fp, topics_d1fp)

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
            probabilities = predict_probabilities_direct_gpu(proba_callback, tf.constant(topic_ids), tf.constant(content_ids),
                                             full_topics_data,
                                             full_contents_data)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            total_probabilities[tlow:thigh] = probabilities.numpy()
            del probabilities
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 6000, max_batch_size)

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

@tf.function
def predict_probabilities_direct_gpu_stepup_dimreduce(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp):
    return proba_callback.predict_probabilities_with_data_return_gpu_dimreduce(topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp)
def obtain_tuple_based_probas_stepup_dimreduce(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp, batch_size=70000):
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
            probabilities = predict_probabilities_direct_gpu_stepup_dimreduce(proba_callback, tf.constant(topic_ids), tf.constant(content_ids),
                                                             full_topics_d1, full_contents_d1,
                                                             full_topics_d1fp, full_contents_d1fp)
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            total_probabilities[tlow:thigh] = probabilities.numpy()
            del probabilities
            # if success we update
            tlow = thigh
            continuous_success += 1
            if continuous_success == 3:
                continuous_success = 0
                batch_size = min(batch_size + 6000, max_batch_size)

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

@tf.function
def predict_probabilities_direct_gpu2(proba_callback, topics_tuple, contents_tuple, full_topics_data, full_contents_data):
    return proba_callback.predict_probabilities_with_data_return_gpu(topics_tuple, contents_tuple, full_topics_data, full_contents_data)

# the chunk is used for buffering. the batch size is dynamic, but it will be always be < chunk size.
def obtain_topic_based_probas(proba_callback, topics_restrict, contents_restrict, topics_folder, out_probs_folder,
                              full_topics_data, full_contents_data, batch_size=70000, chunk_size=8388608):
    assert os.path.exists(topics_folder)
    topics_restrict = np.unique(topics_restrict)
    contents_restrict = np.unique(contents_restrict)
    if not os.path.exists(out_probs_folder):
        os.mkdir(out_probs_folder)

    max_batch_size = chunk_size
    continuous_success = 0

    topics_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers
    contents_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers

    written_into_chunks = 0

    total_probas_write = 0

    for k in range(len(topics_restrict)):
        topic_num_id = topics_restrict[k]
        if os.path.isfile(topics_folder + str(topic_num_id) + ".npy"):
            # has corr, we compute the probas for all.
            content_num_ids = np.load(topics_folder + str(topic_num_id) + ".npy")
        else:
            # no corr, we compute all
            content_num_ids = contents_restrict
        # write into chunk
        topics_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = topic_num_id
        contents_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = content_num_ids
        written_into_chunks = written_into_chunks + len(content_num_ids)

        # use model to predict here
        while written_into_chunks >= batch_size:
            topics_pred_id = topics_tuple_chunk[:batch_size]
            contents_pred_id = contents_tuple_chunk[:batch_size]
            try:
                probabilities = predict_probabilities_direct_gpu2(proba_callback, tf.constant(topics_pred_id),
                                                                  tf.constant(contents_pred_id),
                                                                  full_topics_data, full_contents_data)
            except tf.errors.ResourceExhaustedError as err:
                probabilities = None
            if probabilities is not None:
                probabilities_np = probabilities.numpy()
                np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
                del probabilities, probabilities_np
                total_probas_write += 1

                if written_into_chunks > batch_size:
                    topics_tuple_pending = topics_tuple_chunk[batch_size:written_into_chunks]
                    contents_tuple_pending = contents_tuple_chunk[batch_size:written_into_chunks]
                    written_into_chunks = written_into_chunks - batch_size
                    topics_tuple_chunk[:written_into_chunks] = topics_tuple_pending
                    contents_tuple_chunk[:written_into_chunks] = contents_tuple_pending
                else:
                    written_into_chunks = 0

                gc.collect()
                # if success we update
                continuous_success += 1
                if continuous_success == 3:
                    continuous_success = 0
                    batch_size = min(batch_size + 30000, max_batch_size)
            else:
                batch_size = max(batch_size - 3000, 1)
                max_batch_size = batch_size
                continuous_success = 0
                gc.collect()

        if k % 1000 == 0:
            print("Completed topic " + str(k) + " out of " + str(len(topics_restrict)) + " for probabilities calculation")

    # compute remaining data
    if written_into_chunks > 0:
        tlow = 0
        while tlow < written_into_chunks:
            thigh = min(tlow + batch_size, written_into_chunks)
            topics_pred_id = topics_tuple_chunk[tlow:thigh]
            contents_pred_id = contents_tuple_chunk[tlow:thigh]
            probabilities = predict_probabilities_direct_gpu2(proba_callback, tf.constant(topics_pred_id),
                                                              tf.constant(contents_pred_id),
                                                              full_topics_data, full_contents_data)
            probabilities_np = probabilities.numpy()
            del probabilities

            np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
            del probabilities_np, topics_pred_id, contents_pred_id
            total_probas_write += 1
            tlow = thigh
    del topics_tuple_chunk, contents_tuple_chunk
    gc.collect()
    return total_probas_write

@tf.function
def predict_probabilities_direct_gpu2_stepup_dimreduce(proba_callback, topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp):
    return proba_callback.predict_probabilities_with_data_return_gpu_dimreduce(topics_tuple, contents_tuple, full_topics_d1, full_contents_d1,
                              full_topics_d1fp, full_contents_d1fp)

# the chunk is used for buffering. the batch size is dynamic, but it will be always be < chunk size.
def obtain_topic_based_probas_stepup_dimreduce(proba_callback, topics_restrict, contents_restrict, topics_folder, out_probs_folder,
                                               full_topics_d1, full_contents_d1, full_topics_d1fp, full_contents_d1fp, batch_size=70000, chunk_size=8388608):
    assert os.path.exists(topics_folder)
    topics_restrict = np.unique(topics_restrict)
    contents_restrict = np.unique(contents_restrict)
    if not os.path.exists(out_probs_folder):
        os.mkdir(out_probs_folder)

    max_batch_size = chunk_size
    continuous_success = 0

    topics_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers
    contents_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers

    written_into_chunks = 0

    total_probas_write = 0

    for k in range(len(topics_restrict)):
        topic_num_id = topics_restrict[k]
        if os.path.isfile(topics_folder + str(topic_num_id) + ".npy"):
            # has corr, we compute the probas for all.
            content_num_ids = np.load(topics_folder + str(topic_num_id) + ".npy")
        else:
            # no corr, we compute all
            content_num_ids = contents_restrict
        # write into chunk
        topics_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = topic_num_id
        contents_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = content_num_ids
        written_into_chunks = written_into_chunks + len(content_num_ids)

        # use model to predict here
        while written_into_chunks >= batch_size:
            print("Chunk too large, evaluating and shrinking chunk: ", written_into_chunks, "   Batch size: ", batch_size)
            ctime = time.time()
            topics_pred_id = topics_tuple_chunk[:batch_size]
            contents_pred_id = contents_tuple_chunk[:batch_size]
            try:
                probabilities = predict_probabilities_direct_gpu2_stepup_dimreduce(proba_callback, tf.constant(topics_pred_id),
                                                                  tf.constant(contents_pred_id),
                                                                  full_topics_d1, full_contents_d1, full_topics_d1fp, full_contents_d1fp)
            except tf.errors.ResourceExhaustedError as err:
                probabilities = None
            if probabilities is not None:
                probabilities_np = probabilities.numpy()
                np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
                del probabilities, probabilities_np
                total_probas_write += 1

                if written_into_chunks > batch_size:
                    topics_tuple_pending = topics_tuple_chunk[batch_size:written_into_chunks]
                    contents_tuple_pending = contents_tuple_chunk[batch_size:written_into_chunks]
                    written_into_chunks = written_into_chunks - batch_size
                    topics_tuple_chunk[:written_into_chunks] = topics_tuple_pending
                    contents_tuple_chunk[:written_into_chunks] = contents_tuple_pending
                else:
                    written_into_chunks = 0
                ctime = time.time() - ctime
                print("Shrinked chunk size into ", written_into_chunks, "    Time taken: ", ctime)

                # if success we update
                continuous_success += 1
                if continuous_success == 3:
                    continuous_success = 0
                    batch_size = min(batch_size + 30000, max_batch_size)
            else:
                batch_size = max(batch_size - 30000, 1)
                max_batch_size = batch_size
                continuous_success = 0

            gc.collect()

        if k % 1000 == 0:
            print("Completed topic " + str(k) + " out of " + str(len(topics_restrict)) + " for probabilities calculation")

    # compute remaining data
    if written_into_chunks > 0:
        print("Writing additional data: ", written_into_chunks, " Current batch size: ", batch_size)
        tlow = 0
        while tlow < written_into_chunks:
            thigh = min(tlow + batch_size, written_into_chunks)
            print("Write additional data loop:  ", tlow, "-", thigh)
            ctime = time.time()
            topics_pred_id = topics_tuple_chunk[tlow:thigh]
            contents_pred_id = contents_tuple_chunk[tlow:thigh]
            probabilities = predict_probabilities_direct_gpu2_stepup_dimreduce(proba_callback, tf.constant(topics_pred_id),
                                                              tf.constant(contents_pred_id),
                                                              full_topics_d1, full_contents_d1, full_topics_d1fp, full_contents_d1fp)
            probabilities_np = probabilities.numpy()
            del probabilities

            np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
            del probabilities_np, topics_pred_id, contents_pred_id
            total_probas_write += 1
            tlow = thigh

            ctime = time.time() - ctime
            print("Completed ", tlow, "-", thigh, "   Time taken: ", ctime)
    del topics_tuple_chunk, contents_tuple_chunk
    gc.collect()
    return total_probas_write

@tf.function
def predict_probabilities_direct_gpu2_stepup3_dimreduce(proba_callback, topics_tuple, contents_tuple,
                                                        full_topics_d1os, full_contents_d1os,
                                                        full_topics_d1dp, full_contents_d1dp,
                                                        full_topics_d1fp, full_contents_d1fp):
    return proba_callback.predict_probabilities_with_data_return_gpu_3dimreduce(topics_tuple, contents_tuple,
                                                                        full_topics_d1os, full_contents_d1os,
                                                                        full_topics_d1dp, full_contents_d1dp,
                                                                        full_topics_d1fp, full_contents_d1fp)

# the chunk is used for buffering. the batch size is dynamic, but it will be always be < chunk size.
def obtain_topic_based_probas_stepup_dimreduce3(proba_callback, topics_restrict, contents_restrict, topics_folder, out_probs_folder,
                                                full_topics_d1os, full_contents_d1os,
                                                full_topics_d1dp, full_contents_d1dp,
                                                full_topics_d1fp, full_contents_d1fp,
                                                batch_size=70000, chunk_size=8388608):
    assert os.path.exists(topics_folder)
    topics_restrict = np.unique(topics_restrict)
    contents_restrict = np.unique(contents_restrict)
    if not os.path.exists(out_probs_folder):
        os.mkdir(out_probs_folder)

    max_batch_size = chunk_size
    continuous_success = 0

    topics_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers
    contents_tuple_chunk = np.zeros(shape=chunk_size, dtype=np.int32) # buffers

    written_into_chunks = 0

    total_probas_write = 0

    for k in range(len(topics_restrict)):
        topic_num_id = topics_restrict[k]
        if os.path.isfile(topics_folder + str(topic_num_id) + ".npy"):
            # has corr, we compute the probas for all.
            content_num_ids = np.load(topics_folder + str(topic_num_id) + ".npy")
        else:
            # no corr, we compute all
            content_num_ids = contents_restrict
        # write into chunk
        topics_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = topic_num_id
        contents_tuple_chunk[written_into_chunks:written_into_chunks + len(content_num_ids)] = content_num_ids
        written_into_chunks = written_into_chunks + len(content_num_ids)

        # use model to predict here
        while written_into_chunks >= batch_size:
            print("Chunk too large, evaluating and shrinking chunk: ", written_into_chunks, "   Batch size: ", batch_size)
            ctime = time.time()
            topics_pred_id = topics_tuple_chunk[:batch_size]
            contents_pred_id = contents_tuple_chunk[:batch_size]
            try:
                probabilities = predict_probabilities_direct_gpu2_stepup3_dimreduce(proba_callback,
                                                                tf.constant(topics_pred_id), tf.constant(contents_pred_id),
                                                                full_topics_d1os, full_contents_d1os,
                                                                full_topics_d1dp, full_contents_d1dp,
                                                                full_topics_d1fp, full_contents_d1fp)
            except tf.errors.ResourceExhaustedError as err:
                probabilities = None
            if probabilities is not None:
                probabilities_np = probabilities.numpy()
                np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
                np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
                del probabilities, probabilities_np
                total_probas_write += 1

                if written_into_chunks > batch_size:
                    topics_tuple_pending = topics_tuple_chunk[batch_size:written_into_chunks]
                    contents_tuple_pending = contents_tuple_chunk[batch_size:written_into_chunks]
                    written_into_chunks = written_into_chunks - batch_size
                    topics_tuple_chunk[:written_into_chunks] = topics_tuple_pending
                    contents_tuple_chunk[:written_into_chunks] = contents_tuple_pending
                else:
                    written_into_chunks = 0
                ctime = time.time() - ctime
                print("Shrinked chunk size into ", written_into_chunks, "    Time taken: ", ctime)

                # if success we update
                continuous_success += 1
                if continuous_success == 3:
                    continuous_success = 0
                    batch_size = min(batch_size + 30000, max_batch_size)
            else:
                batch_size = max(batch_size - 30000, 1)
                max_batch_size = batch_size
                continuous_success = 0

            gc.collect()

        if k % 1000 == 0:
            print("Completed topic " + str(k) + " out of " + str(len(topics_restrict)) + " for probabilities calculation")

    # compute remaining data
    if written_into_chunks > 0:
        print("Writing additional data: ", written_into_chunks, " Current batch size: ", batch_size)
        tlow = 0
        while tlow < written_into_chunks:
            thigh = min(tlow + batch_size, written_into_chunks)
            print("Write additional data loop:  ", tlow, "-", thigh)
            ctime = time.time()
            topics_pred_id = topics_tuple_chunk[tlow:thigh]
            contents_pred_id = contents_tuple_chunk[tlow:thigh]
            probabilities = predict_probabilities_direct_gpu2_stepup3_dimreduce(proba_callback,
                                                                tf.constant(topics_pred_id), tf.constant(contents_pred_id),
                                                                full_topics_d1os, full_contents_d1os,
                                                                full_topics_d1dp, full_contents_d1dp,
                                                                full_topics_d1fp, full_contents_d1fp)
            probabilities_np = probabilities.numpy()
            del probabilities

            np.save(out_probs_folder + str(total_probas_write) + "_topics.npy", topics_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_contents.npy", contents_pred_id)
            np.save(out_probs_folder + str(total_probas_write) + "_probas.npy", probabilities_np)
            del probabilities_np, topics_pred_id, contents_pred_id
            total_probas_write += 1
            tlow = thigh

            ctime = time.time() - ctime
            print("Completed ", tlow, "-", thigh, "   Time taken: ", ctime)
    del topics_tuple_chunk, contents_tuple_chunk
    gc.collect()
    return total_probas_write

def obtain_topk_from_probas_folder(topics_restrict, out_probs_folder,
                              total_probas_write, topk = 30):
    next_file = 1
    topics_pred_ids = np.load(out_probs_folder + str(0) + "_topics.npy")
    contents_pred_ids = np.load(out_probs_folder + str(0) + "_contents.npy")
    probas = np.load(out_probs_folder + str(0) + "_probas.npy")


    predictions = []

    for k in range(len(topics_restrict)):
        topic_num_id = topics_restrict[k]
        left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
        right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")
        if right == left:
            topics_pred_ids = np.load(out_probs_folder + str(next_file) + "_topics.npy")
            contents_pred_ids = np.load(out_probs_folder + str(next_file) + "_contents.npy")
            probas = np.load(out_probs_folder + str(next_file) + "_probas.npy")
            next_file += 1
            left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
            right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")

            current_content_ids = contents_pred_ids[left:right]
            current_probas = probas[left:right]
        else:
            current_content_ids = contents_pred_ids[left:right]
            current_probas = probas[left:right]
            
        while right == len(probas) and next_file < total_probas_write:
            topics_pred_ids = np.load(out_probs_folder + str(next_file) + "_topics.npy")
            contents_pred_ids = np.load(out_probs_folder + str(next_file) + "_contents.npy")
            probas = np.load(out_probs_folder + str(next_file) + "_probas.npy")
            next_file += 1

            left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
            right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")

            current_content_ids = np.concatenate([current_content_ids, contents_pred_ids[left:right]], axis=0)
            current_probas = np.concatenate([current_probas, probas[left:right]], axis=0)

        if topk < len(current_content_ids):
            idx = np.argpartition(current_probas, -topk)[-topk:]
            topk_preds = current_content_ids[idx]
        else:
            topk_preds = current_content_ids

        predictions.append(topk_preds)

        if k % 1000 == 0:
            print("Completed topic " + str(k) + " out of " + str(len(topics_restrict)) + " for topk prediction")
    return predictions


def obtain_multiple_topk_from_probas_folder(topics_restrict, out_probs_folder, total_probas_write, topk_multiple):
    next_file = 1
    topics_pred_ids = np.load(out_probs_folder + str(0) + "_topics.npy")
    contents_pred_ids = np.load(out_probs_folder + str(0) + "_contents.npy")
    probas = np.load(out_probs_folder + str(0) + "_probas.npy")

    # for each topk val, save the topk predictions.
    predictions = {}
    for k in topk_multiple:
        predictions[k] = []

    for k in range(len(topics_restrict)):
        topic_num_id = topics_restrict[k]
        left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
        right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")
        if right == left:
            topics_pred_ids = np.load(out_probs_folder + str(next_file) + "_topics.npy")
            contents_pred_ids = np.load(out_probs_folder + str(next_file) + "_contents.npy")
            probas = np.load(out_probs_folder + str(next_file) + "_probas.npy")
            next_file += 1
            left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
            right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")

            current_content_ids = contents_pred_ids[left:right]
            current_probas = probas[left:right]
        else:
            current_content_ids = contents_pred_ids[left:right]
            current_probas = probas[left:right]

        while right == len(probas) and next_file < total_probas_write:
            topics_pred_ids = np.load(out_probs_folder + str(next_file) + "_topics.npy")
            contents_pred_ids = np.load(out_probs_folder + str(next_file) + "_contents.npy")
            probas = np.load(out_probs_folder + str(next_file) + "_probas.npy")
            next_file += 1

            left = np.searchsorted(topics_pred_ids, topic_num_id, side="left")
            right = np.searchsorted(topics_pred_ids, topic_num_id, side="right")

            current_content_ids = np.concatenate([current_content_ids, contents_pred_ids[left:right]], axis=0)
            current_probas = np.concatenate([current_probas, probas[left:right]], axis=0)

        topk = np.max(topk_multiple)
        if topk < len(current_content_ids):
            idx = np.argpartition(current_probas, -topk)[-topk:]

            topk_probas = current_probas[idx]
            topk_preds = current_content_ids[idx]
        else:
            topk_probas = current_probas
            topk_preds = current_content_ids

        idx = np.argsort(topk_probas)
        for topk_spec in topk_multiple:
            if topk_spec < len(topk_preds):
                predictions[topk_spec].append(topk_preds[idx[-topk_spec:]])
            else:
                predictions[topk_spec].append(topk_preds)

        if k % 1000 == 0:
            print("Completed topic " + str(k) + " out of " + str(len(topics_restrict)) + " for topk prediction")
    return predictions