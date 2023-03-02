# this is not the full pipeline. given the vector information (either from BERT or my custom models, output the predicted
# contents and topics)

import numpy as np
import config
import pandas as pd
import time
import math
import gc
import tensorflow as tf

class Node:
    def __init__(self, level, topic_num_id, topic_str_id):
        self.parent = None
        self.children = []
        self.topic_num_id = topic_num_id
        self.topic_str_id = topic_str_id
        self.preorder_id = -1 # the index id in the sense of preorder traversal ordering
        self.subtree_end_id = -1 # the final subnode (not inclusive) in the subtree with self as root

    def __str__(self):
        return "Topic: " + self.topic_str_id + "   " + self.topic_num_id

    def __del__(self):
        for child in self.children:
            del child
        del self.children

def find_node_by_str_id(total_nodes, str_id):
    for node in total_nodes:
        if node.topic_str_id == str_id:
            return node
    return None


def compute_preorder_id(node, cur_id, topic_id_to_preorder_id, preorder_id_to_topic_id, topic_id_to_subtree_end):
    node.preorder_id = cur_id
    topic_id_to_preorder_id[node.topic_num_id] = node.preorder_id
    preorder_id_to_topic_id[node.preorder_id] = node.topic_num_id

    last_id = cur_id
    for child in node.children:
        last_id = compute_preorder_id(child, last_id + 1, topic_id_to_preorder_id, preorder_id_to_topic_id, topic_id_to_subtree_end)
    node.subtree_end_id = last_id + 1

    topic_id_to_subtree_end[node.topic_num_id] = node.subtree_end_id
    return last_id

# topics should be the pandas dataframe for topics.
def generate_tree_structure_information(topics):
    topics_inv_map = pd.Series(data=np.arange(len(topics)), index=topics.index)
    topic_trees = []
    total_nodes = []
    for str_id in topics.loc[topics["level"] == 0].index:
        node = Node(level=0, topic_num_id=topics_inv_map[str_id], topic_str_id=str_id)
        topic_trees.append(node)
        total_nodes.append(node)

    # generate tree structure
    for level in range(np.max(topics["level"].unique())):
        for str_id in topics.loc[topics["level"] == level + 1].index:
            parent = find_node_by_str_id(total_nodes, topics.loc[str_id, "parent"])

            node = Node(level=0, topic_num_id=topics_inv_map[str_id], topic_str_id=str_id)
            node.parent = parent
            parent.children.append(node)

            total_nodes.append(node)

    topic_id_to_preorder_id = np.zeros(shape=len(total_nodes), dtype=np.int32)
    topic_id_to_subtree_end = np.zeros(shape=len(total_nodes), dtype=np.int32)
    preorder_id_to_topic_id = np.zeros(shape=len(total_nodes), dtype=np.int32)

    cur_id = 0
    for node in topic_trees:
        cur_id = compute_preorder_id(node, cur_id, topic_id_to_preorder_id, preorder_id_to_topic_id, topic_id_to_subtree_end) + 1

    del total_nodes, topic_trees
    gc.collect()
    
    return topics_inv_map, topic_id_to_preorder_id, topic_id_to_subtree_end, preorder_id_to_topic_id

# LEGACY CODE, used in previous implementations using GPU

"""def obtain_int32mask(start, end, size):
    mask = np.zeros(shape=int(math.ceil(size / 32.0)), dtype=np.int32)
    mstart = start // 32
    mend = end // 32

    int32start = start % 32
    int32end = end % 32

    if int32start == 0:
        startmask = -1
    else:
        startmask = np.power(2, 32 - int32start, dtype=np.int64) - 1
    if int32end == 0:
        endmask = -1
    else:
        endmask = np.power(2, 32 - int32end, dtype=np.int64) - 1

    if mend == mstart:
        mask[mend] = np.bitwise_xor(np.array([startmask], dtype=np.int32), np.array([endmask], dtype=np.int32))[0]
    else:
        if mend > mstart + 1:
            mask[(mstart + 1):mend] = -1
        mask[mstart] = startmask
        mask[mend] = np.bitwise_not(endmask)
    return mask

bits_mask = np.power(2,np.flip(np.arange(32, dtype = np.int32)))
bits_mask_tf = tf.constant(np.power(2,np.flip(np.arange(32, dtype = np.int32))), tf.int32)
def bool_to_int32mask(bool_mask):
    rm = len(bool_mask) % 32
    if rm == 0:
        padding = 0
    else:
        padding = 32 - rm
    rs = np.reshape(np.pad(bool_mask, pad_width = (0, padding)), newshape = (int(math.ceil(len(bool_mask) / 32.0)), 32))
    return np.dot(rs, bits_mask).astype(np.int32)

def bool_to_int32mask_tf(bool_mask):
    rm = tf.math.floormod(bool_mask.shape[1], 32)
    pad_res = tf.concat([bool_mask, tf.zeros(shape = (bool_mask.shape[0], 32 - rm), dtype=bool_mask.dtype)], axis=1)
    rs = tf.reshape(pad_res, shape=(bool_mask.shape[0], tf.math.floordiv(pad_res.shape[1], 32), 32))
    return tf.linalg.matvec(rs, bits_mask_tf)

def bool_to_int32mask_tf_row(bool_mask):
    rm = tf.math.floormod(bool_mask.shape[0], 32)
    pad_res = tf.concat([bool_mask, tf.zeros(shape = (32 - rm), dtype=bool_mask.dtype)], axis=0)
    rs = tf.reshape(pad_res, shape=(tf.math.floordiv(pad_res.shape[0], 32), 32))
    return tf.linalg.matvec(rs, bits_mask_tf)

@tf.function
def graph_mask_any(res_mask, topic_tree_mask):
    return tf.reduce_any(tf.bitwise.bitwise_and(
                    res_mask,
                    topic_tree_mask
                ) != 0, axis=1)

@tf.function
def res_mask_from_probabilities(probabilities, length_contents, length_topics, accept_threshold, preorder_id_to_topics_restrict_id):
    probabilities = tf.reshape(probabilities, shape=(length_contents, length_topics))
    pad_probs = tf.concat([probabilities, tf.zeros(shape=(probabilities.shape[0], 1), dtype=probabilities.dtype)],
                          axis=1)
    has_cor_places = pad_probs > accept_threshold
    has_correlation_topics_in_preorder = tf.cast(tf.gather(has_cor_places, preorder_id_to_topics_restrict_id, axis=1),
                                                 dtype=tf.int32)

    return bool_to_int32mask_tf(has_correlation_topics_in_preorder)

@tf.function
def graph_mask_any_general(res_mask, topic_tree_mask):
    loop_variables = (tf.constant(0),
                      tf.TensorArray(tf.bool, size=res_mask.shape[0], dynamic_size=False,
                                     element_shape=tf.TensorShape([topic_tree_mask.shape[0]]) ))

    def condition(k, mmasked_result):
        return k < res_mask.shape[0]

    def body(k, mmasked_result):
        print("First eager exec: ", k)
        return (k+1, mmasked_result.write(k, graph_mask_any(topic_tree_mask, res_mask[k, :])))

    return tf.while_loop(condition, body, loop_variables, parallel_iterations=4)[1].stack()

@tf.function
def obtain_masked_result_directly(probabilities, length_contents, length_topics,
                                  accept_threshold, preorder_id_to_topics_restrict_id, topic_tree_mask):
    res_mask = res_mask_from_probabilities(probabilities, length_contents, length_topics, accept_threshold, preorder_id_to_topics_restrict_id)
    return graph_mask_any_general(res_mask, topic_tree_mask)
@tf.function
def obtain_masked_result_directly_with_segsum(probabilities, length_contents, length_topics,
                                  accept_threshold, preorder_id_to_topics_restrict_id, topic_tree_min,
                                  topic_tree_max, parallel_execs):
    has_correlation_topics_in_preorder = preorder_correlations_from_probabilities(probabilities, length_contents, length_topics, accept_threshold, preorder_id_to_topics_restrict_id)
    preorder_length = has_correlation_topics_in_preorder.shape[0]
    loop_variables = (tf.constant(0),
                      tf.TensorArray(tf.bool, size=topic_tree_min.shape[0], dynamic_size=False,
                                     element_shape=tf.TensorShape([length_contents])))

    def condition(k, mmasked_result):
        return k < topic_tree_min.shape[0]

    def body(k, mmasked_result):
        print("First eager exec: ", k)
        rg = tf.range(preorder_length)
        seg_mask = tf.cast(tf.logical_and(rg >= topic_tree_min[k], rg < topic_tree_max[k]), dtype=tf.int32)
        agg = tf.math.unsorted_segment_sum(has_correlation_topics_in_preorder, seg_mask, num_segments=2)[1, :]
        return (k + 1, mmasked_result.write(k,
            agg > 0
        ))

    return tf.transpose(
        tf.while_loop(condition, body, loop_variables,
                      parallel_iterations=parallel_execs
        )[1].stack()
    )

@tf.function
def fast_multiply_has(mat1, mat2):
    return tf.linalg.matmul(mat1, mat2) > 0 """

@tf.function
def preorder_correlations_from_probabilities(probabilities, length_contents, length_topics, accept_threshold, preorder_id_to_topics_restrict_id):
    probabilities = tf.reshape(probabilities, shape=(length_contents, length_topics))
    pad_probs = tf.concat([probabilities, tf.zeros(shape=(probabilities.shape[0], 1), dtype=probabilities.dtype)],
                          axis=1)
    has_cor_places = pad_probs > accept_threshold
    has_correlation_topics_in_preorder = tf.gather(has_cor_places, preorder_id_to_topics_restrict_id, axis=1)
    return has_correlation_topics_in_preorder

@tf.function
def predict_contents(proba_callback, content_ids, topics_restrict, full_topics_data, full_contents_data):
    contents_id = tf.repeat(content_ids, topics_restrict.shape[0])
    topics_id = tf.tile(topics_restrict, [content_ids.shape[0]])

    probabilities = proba_callback.predict_probabilities_with_data_return_gpu(topics_id, contents_id, full_topics_data,
                                                                   full_contents_data)
    return probabilities

# in this case device must be GPU. for each content in contents_restrict, we compute the possible topics the contents
# belong to.
def obtain_contentwise_tree_structure(proba_callback, data_topics, topics_restrict, contents_restrict,
                                      full_topics_data, full_contents_data, accept_threshold = 0.6,
                                      init_batch_size = 10, init_max_batch_size = 30):
    topics_restrict = np.sort(topics_restrict)
    contents_restrict = np.sort(contents_restrict)
    topics_inv_map, topic_id_to_preorder_id, topic_id_to_subtree_end, preorder_id_to_topic_id = generate_tree_structure_information(
        data_topics)

    # this is a mapping from the preorder id to the topics restrict id (meaning preorder_id -> k, where
    # topics_restrict[k] = (preorder_id -> topics_id)).
    preorder_id_to_topics_restrict_id = np.zeros(shape=len(preorder_id_to_topic_id), dtype=np.int32)
    left_side = np.searchsorted(topics_restrict, preorder_id_to_topic_id, side="left")
    right_side = np.searchsorted(topics_restrict, preorder_id_to_topic_id, side="right")

    preorder_id_to_topics_restrict_id[right_side > left_side] = left_side[right_side > left_side]
    preorder_id_to_topics_restrict_id[right_side <= left_side] = len(topics_restrict)
    del left_side, right_side
    preorder_id_to_topics_restrict_id = tf.constant(preorder_id_to_topics_restrict_id)


    # the variable used to store the correlations between topic and content.
    # this are the possible topics per content.
    content_correlations = np.empty(shape = (full_contents_data.shape[0]), dtype = "object")
    for k in range(full_contents_data.shape[0]):
        content_correlations[k] = []

    # now we compute the per content topic trees here.
    ctime = time.time()
    length = len(contents_restrict)

    batch_size = init_batch_size
    max_batch_size = init_max_batch_size
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = 0
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        content_ids = contents_restrict[np.arange(tlow, thigh)]
        try:
            probabilities = predict_contents(proba_callback, tf.constant(content_ids), tf.constant(topics_restrict), full_topics_data,
                                         full_contents_data)
            preorder_probas = preorder_correlations_from_probabilities(probabilities, thigh-tlow, len(topics_restrict),
                                                   accept_threshold, preorder_id_to_topics_restrict_id)
        except tf.errors.ResourceExhaustedError as err:
            preorder_probas = None
            # masked_result = None
            # res_mask = None
            # probabilities = None
        # if probabilities is not None:
        if preorder_probas is not None:
            gc.collect()

            probas_np = preorder_probas.numpy()
            del preorder_probas
            masked_result = np.zeros(shape=(len(data_topics), thigh-tlow), dtype=np.bool)
            for k in range(len(data_topics)):
                masked_result[k,:] = np.any(probas_np[:, topic_id_to_preorder_id[k]:topic_id_to_subtree_end[k]], axis=1)

            for k in range(tlow, thigh):
                # masked_result = graph_mask_any(res_mask[k - tlow, :], topic_tree_mask)
                content_correlations[contents_restrict[k]] = list(np.where(masked_result[:, k-tlow])[0])
            del masked_result
            del preorder_probas, probabilities
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

    # create this by topic.
    topic_correlations = np.empty(shape = (full_topics_data.shape[0]), dtype = "object")
    for k in range(full_topics_data.shape[0]):
        topic_correlations[k] = []
    for k in range(len(contents_restrict)):
        for top_num_id in contents_restrict[k]:
            topic_correlations[top_num_id].append(contents_restrict[k])
    return content_correlations, topic_correlations
