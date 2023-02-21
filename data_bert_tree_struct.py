import copy
import data_bert
import numpy as np
import pandas as pd
import os
import config
import time


# compute and obtain the tree structure of topics.
levels = 10

topics_group = [] # topics_group[k] are the partition by kth level
# each topics_group[k], is a dict {"groups":groups, "group_ids":group_ids}.
# groups and group_ids are np arrays with the same size.
# each group_ids[j] correspond to the integer id for the topic at the kth level
# each group[j] is another np array containing all the integer ids of subtopics (and itself)


topic_trees_contents = [] # topic_trees_contents is a list, where each topic[k] is list of tuples (represented by two
# np arrays topic[k]["topic_id_klevel"], topic[k]["content_id"]), containing the correlations of topics and contents,
# sorted first by topics and then by contents. In each tuple (topic, content) in topic[k], the topic corresponds to the
# topicID in kth level (which means topics_group[k]["group_ids"][topic] would be the topic_num_id for the global topics),
# and content would directly be the content_num_id for the global contents list.
topic_trees_contents_train = [] # same thing, just restricted to train set.
topic_trees_contents_test = [] # same thing, just restricted to test set.


proximity_structure = [] # list of dicts to store the proximity structure.
further_proximity_structure = [] # list of dicts to store the further proximity structure.

# same format as data_bert.has_correlation..... these store the tuples topics-contents such that the topics / contents
# are "close" to each other, in the sense that they are either parent/child, siblings, or grandparents/grandchildren (distance 2)
has_close_correlation_topics = []
has_close_correlation_contents = []
has_close_correlation_train_topics = []
has_close_correlation_train_contents = []
has_close_correlation_test_topics = []
has_close_correlation_test_contents = []

has_further_correlation_topics = []
has_further_correlation_contents = []
has_further_correlation_train_topics = []
has_further_correlation_train_contents = []
has_further_correlation_test_topics = []
has_further_correlation_test_contents = []

def generate_topics_grouping_info():
    # generate the groups
    for k in range(levels + 1):
        topics_k_level = data_bert.topics.loc[data_bert.topics["level"] == k]
        group_ids = np.array(data_bert.topics_inv_map.loc[topics_k_level.index], dtype = np.int32)
        # groups = np.array([np.array([idx], dtype = np.int32) for idx in group_ids], dtype = "object")
        # this cannot be used since numpy somehow converts the dtype of inner arrays to object too. we need it to be int
        groups = np.empty(len(group_ids), dtype="object")
        for tidx in range(len(group_ids)):
            groups[tidx] = np.array([group_ids[tidx]], dtype=np.int32)
        topics_group.append({"groups":groups, "group_ids":group_ids})

        if k > 0:
            topics_i_level = topics_k_level
            for j in range(k):
                i = k-j-1
                parent_idx = pd.Index(list(topics_i_level["parent"]))
                topics_i_level = data_bert.topics.loc[parent_idx]
                par_locs = np.array(data_bert.topics_inv_map.loc[parent_idx], dtype = np.int32)
                par_locs_in_topics = np.searchsorted(topics_group[i]["group_ids"], par_locs)
                for pl_i in range(len(par_locs_in_topics)):
                    pl = par_locs_in_topics[pl_i]
                    old_arr = topics_group[i]["groups"][pl]
                    topics_group[i]["groups"][pl] = np.concatenate([old_arr, np.array([group_ids[pl_i]], dtype = np.int32)])

                    del old_arr
def generate_topics_contents_correlations():
    for plevel in range(len(topics_group)):
        emptydict = {"topic_id_klevel": [], "content_id": []}
        topic_trees_contents.append(emptydict)
    for k in range(len(data_bert.topics)):
        if data_bert.topics.loc[data_bert.topics.index[k], "has_content"]:
            content_ids = data_bert.correlations.loc[data_bert.topics.index[k], "content_ids"].split()
            content_num_ids = list(np.sort(data_bert.contents_inv_map[content_ids].to_numpy()))

            level = data_bert.topics.loc[data_bert.topics.index[k], "level"]
            for plevel in range(level, -1, -1):
                klevel_id = np.searchsorted(topics_group[plevel]["group_ids"], k) # find the location such that
                # topics_group[plevel]["group_ids"][klevel_id] = k
                topic_trees_contents[plevel]["topic_id_klevel"].extend([klevel_id] * len(content_ids))
                topic_trees_contents[plevel]["content_id"].extend(content_num_ids)
    for plevel in range(len(topics_group)):
        topic_ids_kl = topic_trees_contents[plevel]["topic_id_klevel"]
        gcontent_id = topic_trees_contents[plevel]["content_id"]
        topic_trees_contents[plevel]["topic_id_klevel"] = np.array(topic_ids_kl, dtype = np.int32)
        topic_trees_contents[plevel]["content_id"] = np.array(gcontent_id, dtype=np.int32)

def initialize_proximity_structure(prox_struct):
    for k in range(len(data_bert.topics)):
        prox_struct.append({"parent":None, "children":[], "close_prox":[]})

def expand_proximity_structure(prox_struct):
    # do deep copy into np.arrays
    proximity_structure_fix_children = []
    proximity_structure_fix_close_prox = np.empty(len(data_bert.topics), dtype = "object")
    proximity_structure_fix_parent = np.empty(len(data_bert.topics), dtype = np.int32)
    proximity_structure_fix_parent[:] = -1
    for k in range(len(data_bert.topics)):
        proximity_structure_fix_children.append(np.sort(np.array(prox_struct[k]["children"], dtype = np.int32)))
        proximity_structure_fix_close_prox[k] = np.sort(np.array(prox_struct[k]["close_prox"], dtype = np.int32))
        if prox_struct[k]["parent"] is not None:
            proximity_structure_fix_parent[k] = prox_struct[k]["parent"]
        else:
            proximity_structure_fix_parent[k] = -1
    # compute additionals here
    for k in range(len(data_bert.topics)):
        prox_childs = np.unique(np.concatenate([proximity_structure_fix_children[proxies] for proxies in proximity_structure_fix_close_prox[k]]))
        prox_parents = proximity_structure_fix_parent[proximity_structure_fix_close_prox[k]]
        prox_parents = prox_parents[prox_parents != -1]
        prox_struct[k]["close_prox"] = list(np.unique(np.concatenate([prox_childs, prox_parents, proximity_structure_fix_close_prox[k]])))

def generate_proximity_structure(prox_struct, distance = 2):
    initialize_proximity_structure(prox_struct)
    for k in range(len(data_bert.topics)):
        prox_struct[k]["close_prox"].append(k)
        parent = data_bert.topics.loc[data_bert.topics.index[k], "parent"]
        if type(parent) == str:
            parent_num_id = data_bert.topics_inv_map.loc[parent]
            prox_struct[parent_num_id]["children"].append(k)
            prox_struct[k]["parent"] = parent_num_id
    for count in range(distance):
        expand_proximity_structure(prox_struct)

# assumes array_to_search and sorted_values are sorted, and sorted_values are distinct.
# returns all locations k such that array_to_search[k] == sorted_values[j] for some arbitrary j.
def get_rep_indices(array_to_search, sorted_values):
    assert np.all(array_to_search[:-1] <= array_to_search[1:])
    assert np.all(sorted_values[:-1] < sorted_values[1:])
    cp_left = np.searchsorted(array_to_search, sorted_values, side="left")
    cp_right = np.searchsorted(array_to_search, sorted_values, side="right")
    cp_with_contents = cp_right > cp_left
    if(cp_with_contents.astype(np.int32).sum()) == 0:
        return np.array([], dtype = np.int32)
    cp_left = cp_left[cp_with_contents]
    cp_right = cp_right[cp_with_contents]
    ref_total_index = np.arange(cp_left[0], cp_right[-1]+1)
    # find the indices such that they are in [cp_left[j], cp_right[j]) for some j
    ti_locs = (np.searchsorted(cp_left, ref_total_index, side="right") -1) == np.searchsorted(cp_right, ref_total_index, side="right")
    return ref_total_index[ti_locs]

def generate_proximity_correlations():
    global has_close_correlation_topics, has_close_correlation_contents, has_close_correlation_train_topics, has_close_correlation_train_contents
    global has_close_correlation_test_topics, has_close_correlation_test_contents
    ref_total_index = np.arange(len(data_bert.topics))
    for k in range(len(data_bert.topics)):
        close_proxes = np.sort(np.array(proximity_structure[k]["close_prox"], dtype = np.int32))
        close_proxes_locs = get_rep_indices(data_bert.has_correlation_topics, close_proxes)

        content_ids = np.unique(data_bert.has_correlation_contents[close_proxes_locs])
        has_close_correlation_topics.extend([k] * len(content_ids))
        has_close_correlation_contents.extend(list(content_ids))

        if k in data_bert.train_topics_num_id:
            close_proxes_locs = get_rep_indices(data_bert.has_correlation_train_topics, close_proxes)

            content_ids = np.unique(data_bert.has_correlation_train_contents[close_proxes_locs])
            has_close_correlation_train_topics.extend([k] * len(content_ids))
            has_close_correlation_train_contents.extend(list(content_ids))

        if k in data_bert.test_topics_num_id:
            close_proxes_locs = get_rep_indices(data_bert.has_correlation_test_topics, close_proxes)

            content_ids = np.unique(data_bert.has_correlation_test_contents[close_proxes_locs])
            has_close_correlation_test_topics.extend([k] * len(content_ids))
            has_close_correlation_test_contents.extend(list(content_ids))

    has_close_correlation_topics = np.array(has_close_correlation_topics, np.int32)
    has_close_correlation_contents = np.array(has_close_correlation_contents, np.int32)
    has_close_correlation_train_topics = np.array(has_close_correlation_train_topics, np.int32)
    has_close_correlation_train_contents = np.array(has_close_correlation_train_contents, np.int32)
    has_close_correlation_test_topics = np.array(has_close_correlation_test_topics, np.int32)
    has_close_correlation_test_contents = np.array(has_close_correlation_test_contents, np.int32)

def generate_further_proximity_correlations():
    global has_further_correlation_topics, has_further_correlation_contents, has_further_correlation_train_topics, has_further_correlation_train_contents
    global has_further_correlation_test_topics, has_further_correlation_test_contents
    ref_total_index = np.arange(len(data_bert.topics))
    for k in range(len(data_bert.topics)):
        further_proxes = np.sort(np.array(further_proximity_structure[k]["close_prox"], dtype = np.int32))
        further_proxes_locs = get_rep_indices(data_bert.has_correlation_topics, further_proxes)

        content_ids = np.unique(data_bert.has_correlation_contents[further_proxes_locs])
        has_further_correlation_topics.extend([k] * len(content_ids))
        has_further_correlation_contents.extend(list(content_ids))

        if k in data_bert.train_topics_num_id:
            further_proxes_locs = get_rep_indices(data_bert.has_correlation_train_topics, further_proxes)

            content_ids = np.unique(data_bert.has_correlation_train_contents[further_proxes_locs])
            has_further_correlation_train_topics.extend([k] * len(content_ids))
            has_further_correlation_train_contents.extend(list(content_ids))

        if k in data_bert.test_topics_num_id:
            further_proxes_locs = get_rep_indices(data_bert.has_correlation_test_topics, further_proxes)

            content_ids = np.unique(data_bert.has_correlation_test_contents[further_proxes_locs])
            has_further_correlation_test_topics.extend([k] * len(content_ids))
            has_further_correlation_test_contents.extend(list(content_ids))

    has_further_correlation_topics = np.array(has_further_correlation_topics, np.int32)
    has_further_correlation_contents = np.array(has_further_correlation_contents, np.int32)
    has_further_correlation_train_topics = np.array(has_further_correlation_train_topics, np.int32)
    has_further_correlation_train_contents = np.array(has_further_correlation_train_contents, np.int32)
    has_further_correlation_test_topics = np.array(has_further_correlation_test_topics, np.int32)
    has_further_correlation_test_contents = np.array(has_further_correlation_test_contents, np.int32)

generate_topics_grouping_info()
generate_topics_contents_correlations()
generate_proximity_structure(proximity_structure, distance = 2)
generate_proximity_structure(further_proximity_structure, distance = 3)

if os.path.isdir(config.resources_path + "data_bert_tree/"):
    has_close_correlation_topics = np.load(config.resources_path + "data_bert_tree/has_close_correlation_topics.npy")
    has_close_correlation_contents = np.load(config.resources_path + "data_bert_tree/has_close_correlation_contents.npy")
    has_close_correlation_train_topics = np.load(config.resources_path + "data_bert_tree/has_close_correlation_train_topics.npy")
    has_close_correlation_train_contents = np.load(config.resources_path + "data_bert_tree/has_close_correlation_train_contents.npy")
    has_close_correlation_test_topics = np.load(config.resources_path + "data_bert_tree/has_close_correlation_test_topics.npy")
    has_close_correlation_test_contents = np.load(config.resources_path + "data_bert_tree/has_close_correlation_test_contents.npy")
else:
    os.mkdir(config.resources_path + "data_bert_tree/")
    generate_proximity_correlations()
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_topics.npy", has_close_correlation_topics)
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_contents.npy", has_close_correlation_contents)
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_train_topics.npy", has_close_correlation_train_topics)
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_train_contents.npy", has_close_correlation_train_contents)
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_test_topics.npy", has_close_correlation_test_topics)
    np.save(config.resources_path + "data_bert_tree/has_close_correlation_test_contents.npy", has_close_correlation_test_contents)

if os.path.isdir(config.resources_path + "data_bert_tree_further/"):
    has_further_correlation_topics = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_topics.npy")
    has_further_correlation_contents = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_contents.npy")
    has_further_correlation_train_topics = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_train_topics.npy")
    has_further_correlation_train_contents = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_train_contents.npy")
    has_further_correlation_test_topics = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_test_topics.npy")
    has_further_correlation_test_contents = np.load(config.resources_path + "data_bert_tree_further/has_further_correlation_test_contents.npy")
else:
    os.mkdir(config.resources_path + "data_bert_tree_further/")
    generate_further_proximity_correlations()
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_topics.npy", has_further_correlation_topics)
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_contents.npy", has_further_correlation_contents)
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_train_topics.npy", has_further_correlation_train_topics)
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_train_contents.npy", has_further_correlation_train_contents)
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_test_topics.npy", has_further_correlation_test_topics)
    np.save(config.resources_path + "data_bert_tree_further/has_further_correlation_test_contents.npy", has_further_correlation_test_contents)


def has_close_correlations(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_close_correlation_contents, has_close_correlation_topics)

def has_close_correlations_train(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_close_correlation_train_contents, has_close_correlation_train_topics)

def has_close_correlations_test(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_close_correlation_test_contents, has_close_correlation_test_topics)

def has_further_correlations(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_further_correlation_contents, has_further_correlation_topics)

def has_further_correlations_train(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_further_correlation_train_contents, has_further_correlation_train_topics)

def has_further_correlations_test(content_num_ids, topic_num_ids):
    return data_bert.has_correlations_general(content_num_ids, topic_num_ids, has_further_correlation_test_contents, has_further_correlation_test_topics)

def obtain_train_sample(one_sample_size, zero_sample_size):
    return data_bert.obtain_general_sample(one_sample_size, zero_sample_size, has_close_correlation_train_contents, has_close_correlation_train_topics, data_bert.train_contents_num_id, data_bert.train_topics_num_id)

def obtain_test_sample(one_sample_size, zero_sample_size):
    return data_bert.obtain_general_sample(one_sample_size, zero_sample_size, has_close_correlation_test_contents, has_close_correlation_test_topics, data_bert.test_contents_num_id, data_bert.test_topics_num_id)
def obtain_train_square_sample(sample_size):
    return data_bert.obtain_general_square_sample(sample_size, has_close_correlation_train_contents, has_close_correlation_train_topics, data_bert.train_contents_num_id, data_bert.train_topics_num_id)
def obtain_test_square_sample(sample_size):
    return data_bert.obtain_general_square_sample(sample_size, has_close_correlation_test_contents, has_close_correlation_test_topics, data_bert.test_contents_num_id, data_bert.test_topics_num_id)


def obtain_further_train_sample(one_sample_size, zero_sample_size):
    return data_bert.obtain_general_sample(one_sample_size, zero_sample_size, has_further_correlation_train_contents, has_further_correlation_train_topics, data_bert.train_contents_num_id, data_bert.train_topics_num_id)

def obtain_further_test_sample(one_sample_size, zero_sample_size):
    return data_bert.obtain_general_sample(one_sample_size, zero_sample_size, has_further_correlation_test_contents, has_further_correlation_test_topics, data_bert.test_contents_num_id, data_bert.test_topics_num_id)
def obtain_further_train_square_sample(sample_size):
    return data_bert.obtain_general_square_sample(sample_size, has_further_correlation_train_contents, has_further_correlation_train_topics, data_bert.train_contents_num_id, data_bert.train_topics_num_id)
def obtain_further_test_square_sample(sample_size):
    return data_bert.obtain_general_square_sample(sample_size, has_further_correlation_test_contents, has_further_correlation_test_topics, data_bert.test_contents_num_id, data_bert.test_topics_num_id)