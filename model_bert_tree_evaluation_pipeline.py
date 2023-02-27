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
        last_id = compute_preorder_id(child, last_id + 1)
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

def obtain_int32mask(start, end, size):
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
bits_mask = np.power(2,np.flip(np.arange(32, dtype = np.int32)))
def bool_to_int32mask(bool_mask):
    rm = len(bool_mask) % 32
    if rm == 0:
        padding = 0
    else:
        padding = 32 - rm
    rs = np.reshape(np.pad(bool_mask, pad_width = (0, padding)), newshape = (int(math.ceil(len(bool_mask) / 32.0)), 32))
    return np.dot(rs, bits_mask).astype(np.int32)

def predict_contents(proba_callback, content_ids, topics_restrict, full_topics_data, full_contents_data):
    contents_id = tf.repeat(tf.constant(content_ids), len(topics_restrict))
    topics_id = tf.tile(tf.constant(topics_restrict), [len(content_ids)])

    probabilities = proba_callback.predict_probabilities_with_data(topics_id, contents_id, full_topics_data,
                                                                   full_contents_data, "gpu")
    return probabilities

# in this case device must be GPU. for each content in contents_restrict, we compute the possible topics the contents
# belong to.
def obtain_contentwise_tree_structure(proba_callback, data_topics, topics_restrict, contents_restrict,
                                      full_topics_data, full_contents_data, accept_threshold = 0.6):
    topics_inv_map, topic_id_to_preorder_id, topic_id_to_subtree_end, preorder_id_to_topic_id = generate_tree_structure_information(
        data_topics)

    # per each topic (no matter in topics_restrict or not), we create a mask to represent the subtree starting with
    # the topic node. this mask would be in bit format, represented by int32 tensor.
    # the first axis would be the topics axis (in usual topic_num_id), while the second axis is the bitwise axis
    # in topic_preorder_id. the bit at the (topic_num_id, topic_preorder_id) position will signify whether the
    # topic represented by topic_num_id is a (non-strict) ancestor or the topic represented by topic_preorder_id.
    preorder_size = int(math.ceil(len(data_topics) / 32.0))
    topic_tree_mask = np.zeros(shape = (len(data_topics), preorder_size), dtype = np.int32)
    for topic_num_id in range(len(data_topics)):
        topic_tree_mask[topic_num_id, :] = obtain_int32mask(topic_id_to_preorder_id[topic_num_id],
                                                            topic_id_to_subtree_end[topic_num_id], len(data_topics))
    topic_tree_mask = tf.constant(topic_tree_mask, dtype = tf.int32)

    # the variable used to store the correlations between topic and content.
    # this are the possible topics per content.
    content_correlations = np.empty(shape = (full_contents_data.shape[0]), dtype = "object")

    # now we compute the per content topic trees here.
    ctime = time.time()
    length = len(contents_restrict)

    batch_size = 10
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = 0
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        content_ids = contents_restrict[np.arange(tlow, thigh)]
        try:
            probabilities = predict_contents(proba_callback, content_ids, topics_restrict, full_topics_data,
                                         full_contents_data)
            probabilities = (probabilities.reshape((thigh - tlow), len(topics_restrict))) > accept_threshold
        except tf.errors.ResourceExhaustedError as err:
            probabilities = None
        if probabilities is not None:
            gc.collect()
            has_topics_bool_mask = np.zeros(shape=len(data_topics), dtype = np.int32)
            for k in range(tlow, thigh):
                has_correlation_topics_in_preorder = topic_id_to_preorder_id[topics_restrict[probabilities[k, :]]]
                has_topics_bool_mask[:] = 0
                has_topics_bool_mask[has_correlation_topics_in_preorder] = 1
                has_topics_int32_mask = bool_to_int32mask(has_topics_bool_mask)
                has_topics_int32_mask = tf.repeat(tf.expand_dims(tf.constant(has_topics_int32_mask, dtype = tf.int32),
                                                                 axis = 0), len(data_topics), axis = 0)
                masked_result = tf.reduce_any(tf.bitwise.bitwise_and(topic_tree_mask, has_topics_int32_mask) != 0,
                                        axis = 1)
                content_correlations[contents_restrict[k]] = list(np.where(masked_result.numpy())[0])

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
