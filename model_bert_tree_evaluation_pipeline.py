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


# in this case device must be GPU. for each content in contents_restrict, we compute the possible topics the contents
# belong to.
def obtain_contentwise_tree_structure(proba_callback, data_topics, topics_restrict, contents_restrict,
                                      full_topics_data, full_contents_data):
    topics_inv_map, topic_id_to_preorder_id, topic_id_to_subtree_end, preorder_id_to_topic_id = generate_tree_structure_information(
        data_topics)

    # per each topic (no matter in topics_restrict or not), we create a mask to represent the subtree starting with
    # the topic node. this mask would be in bit format, represented by int32 tensor.
    # the first axis would be the topics axis (in usual topic_num_id), while the second axis is the bitwise axis
    # in topic_preorder_id. the bit at the (topic_num_id, topic_preorder_id) position will signify whether the
    # topic represented by topic_num_id is a (non-strict) ancestor or the topic represented by topic_preorder_id.

    # dict of np arrays, where each np array is len(topics_restrict) x topk_values[i], where each row contains the topk predictions
    topk_preds = {}
    for i in range(len(topk_values)):
        topk_preds[topk_values[i]] = np.zeros(shape=(len(topics_restrict), topk_values[i]))

    ctime = time.time()

    length = len(topics_restrict)
    prevlnumber = 0
    max_topk = np.max(topk_values)

    batch_size = greedy_multiple_rows
    tlow = 0
    continuous_success = 0
    prev_tlow = 0
    ctime = 0
    while tlow < length:
        thigh = min(tlow + batch_size, length)
        topic_id_rows = topics_restrict[np.arange(tlow, thigh)]
        try:
            probabilities = predict_rows(proba_callback, topic_id_rows, contents_restrict, full_topics_data,
                                         full_contents_data, device=device)
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
