import config
import numpy as np
import pandas as pd
import json
import gc

contents = pd.read_csv(config.resources_path + "contents_translate.csv", index_col = 0)
correlations = pd.read_csv(config.resources_path + "correlations.csv", index_col = 0)
topics = pd.read_csv(config.resources_path + "topics_translate.csv", index_col = 0)

# inverse map, to obtain the integer id of topic / content given the str id.
contents_inv_map = pd.Series(data = np.arange(len(contents)), index = contents.index)
topics_inv_map = pd.Series(data = np.arange(len(topics)), index = topics.index)

"""# compute and obtain the tree structure of topics.
levels = 10

topics_group = [] # topics_group[k] are the partition by kth level
# each topics_group[k], is a dict {"groups":groups, "group_ids":group_ids}.
# groups and group_ids are np arrays with the same size.
# each group_ids[j] correspond to the integer id for the topic at the kth level
# each group[j] is another np array containing all the integer ids of subtopics (and itself)


topic_trees_contents = [] # topic_trees_contents is a list, where each topic[k] is a 

def generate_topics_grouping_info():
    # generate the groups
    for k in range(levels + 1):
        topics_k_level = topics.loc[topics["level"] == k]
        group_ids = np.array(topics_inv_map.loc[topics_k_level.index], dtype = np.int32)
        groups = np.array([np.array([idx]) for idx in group_ids] ,dtype = "object")
        topics_group.append({"groups":groups, "group_ids":group_ids})

        if k > 0:
            topics_i_level = topics_k_level
            for j in range(k):
                i = k-j-1
                parent_idx = pd.Index(list(topics_i_level["parent"]))
                topics_i_level = topics.loc[parent_idx]
                par_locs = np.array(topics_inv_map.loc[parent_idx], dtype = np.int32)
                par_locs_in_topics = np.searchsorted(topics_group[i]["group_ids"], par_locs)
                for pl_i in range(len(par_locs_in_topics)):
                    pl = par_locs_in_topics[pl_i]
                    old_arr = topics_group[i]["groups"][pl]
                    topics_group[i]["groups"][pl] = np.concatenate([old_arr, np.array([group_ids[pl_i]], dtype = np.int32)])
                    
                    del old_arr

    # generate the contents for each given group
    topics_group[0]

generate_topics_grouping_info()"""
with open(config.resources_path + "temp_channel_components.json") as file:
    channel_components = json.loads(file.read())
with open(config.resources_path + "temp_topic_trees_contents.json") as file:
    topic_trees_contents = json.loads(file.read())

# -------------------------------- restrict the topics and contents to those that have at least 1 description --------------------------------
contents_availability = np.zeros(shape = (len(contents), 3), dtype = bool)
contents_availability[:, 0] = np.load(config.resources_path + "bert_tokens/contents_description/has_content_mask.npy")
contents_availability[:, 1] = np.load(config.resources_path + "bert_tokens/contents_title/has_content_mask.npy")
contents_availability[:, 2] = np.logical_or(contents_availability[:, 0], contents_availability[:, 1])
contents_availability = pd.DataFrame(data = contents_availability, columns = ["description", "title", "some_available"], index = contents.index)

topics_availability = np.zeros(shape = (len(topics), 3), dtype = bool)
topics_availability[:, 0] = np.load(config.resources_path + "bert_tokens/topics_description/has_content_mask.npy")
topics_availability[:, 1] = np.load(config.resources_path + "bert_tokens/topics_title/has_content_mask.npy")
topics_availability[:, 2] = np.logical_or(topics_availability[:, 0], topics_availability[:, 1])
topics_availability = pd.DataFrame(data = topics_availability, columns = ["description", "title", "some_available"], index = topics.index)

learnable_contents = contents_availability.loc[contents_availability["some_available"]].index
learnable_topics = topics_availability.loc[topics_availability["some_available"]].index

# -------------------------------- obtain train test split here --------------------------------
def filter_channel_topics(channels_list, restriction_index = None):
    total_indices = pd.Index([])
    for channel_id in channels_list:
        if restriction_index is not None:
            total_indices = total_indices.append(topics.loc[topics["channel"] == channel_id].index.intersection(restriction_index))
        else:
            total_indices = total_indices.append(topics.loc[topics["channel"] == channel_id].index)
    return total_indices

def filter_channel_contents(channels_list, restriction_index = None):
    total_indices = pd.Index([])
    for channel_id in channels_list:
        additional = pd.Index(topic_trees_contents[channel_id])
        if restriction_index is not None:
            total_indices = total_indices.append( additional.intersection(restriction_index) )
        else:
            total_indices = total_indices.append( additional )
    return total_indices

def index_is_usable(idx):
    contents_count = len(filter_channel_contents(channel_components[idx], learnable_contents))
    topics_count = len(filter_channel_topics(channel_components[idx], learnable_topics))
    return contents_count > 100 and topics_count > 100

train_contents = filter_channel_contents(channel_components[0], learnable_contents)
train_topics = filter_channel_topics(channel_components[0], learnable_topics)

test_data_channels = []
for idx in range(1, len(channel_components)):
    if index_is_usable(idx):
        test_data_channels.extend(channel_components[idx])
test_contents = filter_channel_contents(test_data_channels, learnable_contents)
test_topics = filter_channel_topics(test_data_channels, learnable_topics)

# reorder train test split into natural order
train_contents_num_id = np.sort(contents_inv_map[train_contents].to_numpy())
train_topics_num_id = np.sort(topics_inv_map[train_topics].to_numpy())
test_contents_num_id = np.sort(contents_inv_map[test_contents].to_numpy())
test_topics_num_id = np.sort(topics_inv_map[test_topics].to_numpy())

train_contents = contents.index[train_contents_num_id]
train_topics = topics.index[train_topics_num_id]
test_contents = contents.index[test_contents_num_id]
test_topics = topics.index[test_topics_num_id]

# -------------------------------- create association matrix from correlations --------------------------------
# a list of tuples (topic, content) such that there exists a correlation between them
def fast_contains(sorted_arr, val):
    idx = np.searchsorted(sorted_arr, val, side = "left")
    return idx != len(sorted_arr) and sorted_arr[idx] == val

def fast_contains_multi(sorted_arr, vals):
    return np.searchsorted(sorted_arr, vals, side = "right") > np.searchsorted(sorted_arr, vals, side = "left")

has_correlation_topics = []
has_correlation_contents = []
has_correlation_train_topics = []
has_correlation_train_contents = []
has_correlation_test_topics = []
has_correlation_test_contents = []
for k in range(len(topics)):
    if topics.loc[topics.index[k], "has_content"]:
        content_ids = correlations.loc[topics.index[k], "content_ids"].split()
        has_correlation_topics.extend([k] * len(content_ids))
        has_correlation_contents.extend(list( np.sort(contents_inv_map[content_ids].to_numpy()) ))

        if k in train_topics_num_id:
            """to_add = []
            for content_num_id in contents_inv_map[content_ids]:
                if fast_contains(train_contents_num_id, content_num_id):
                    to_add.append(content_num_id)
            has_correlation_train_topics.extend([k] * len(to_add))
            to_add.sort()
            has_correlation_train_contents.extend(to_add)"""
            to_add = np.array(contents_inv_map[content_ids], dtype = np.int32)
            mask = fast_contains_multi(train_contents_num_id, to_add)
            to_add = np.sort(to_add[mask])
            has_correlation_train_topics.extend([k] * len(to_add))
            has_correlation_train_contents.extend(list(to_add))


        if k in test_topics_num_id:
            """to_add = []
            for content_num_id in contents_inv_map[content_ids]:
                if fast_contains(test_contents_num_id, content_num_id):
                    to_add.append(content_num_id)
            has_correlation_test_topics.extend([k] * len(to_add))
            to_add.sort()
            has_correlation_test_contents.extend(to_add)"""
            to_add = np.array(contents_inv_map[content_ids], dtype=np.int32)
            mask = fast_contains_multi(test_contents_num_id, to_add)
            to_add = np.sort(to_add[mask])
            has_correlation_test_topics.extend([k] * len(to_add))
            has_correlation_test_contents.extend(list(to_add))


has_correlation_contents = np.array(has_correlation_contents, dtype = np.int32)
has_correlation_topics = np.array(has_correlation_topics, dtype = np.int32)
has_correlation_train_contents = np.array(has_correlation_train_contents, dtype = np.int32)
has_correlation_train_topics = np.array(has_correlation_train_topics, dtype = np.int32)
has_correlation_test_contents = np.array(has_correlation_test_contents, dtype = np.int32)
has_correlation_test_topics = np.array(has_correlation_test_topics, dtype = np.int32)

def has_correlation(content_num_id, topic_num_id):
    left = np.searchsorted(has_correlation_topics, topic_num_id, side = "left")
    right = np.searchsorted(has_correlation_topics, topic_num_id, side = "right")

    if left == len(has_correlation_topics) or has_correlation_topics[left] != topic_num_id:
        return False

    subcontents = has_correlation_contents[left:right]
    left = np.searchsorted(subcontents, content_num_id)
    return left != len(subcontents) and subcontents[left] == content_num_id

# same function as above, except we accept a list of points.
def has_correlations(content_num_ids, topic_num_ids):
    assert len(content_num_ids) == len(topic_num_ids)

    left = np.searchsorted(has_correlation_topics, topic_num_ids, side = "left")
    right = np.searchsorted(has_correlation_topics, topic_num_ids, side="right")

    resarr = np.zeros(shape = (len(content_num_ids)), dtype = bool)

    for k in range(len(content_num_ids)):
        subcontents = has_correlation_contents[left[k]:right[k]]
        mleft = np.searchsorted(subcontents, content_num_ids[k])
        resarr[k] = mleft != len(subcontents) and subcontents[mleft] == content_num_ids[k]
    return resarr

def has_correlations_train(content_num_ids, topic_num_ids):
    assert len(content_num_ids) == len(topic_num_ids)

    left = np.searchsorted(has_correlation_train_topics, topic_num_ids, side = "left")
    right = np.searchsorted(has_correlation_train_topics, topic_num_ids, side="right")

    resarr = np.zeros(shape = (len(content_num_ids)), dtype = bool)

    for k in range(len(content_num_ids)):
        subcontents = has_correlation_train_contents[left[k]:right[k]]
        mleft = np.searchsorted(subcontents, content_num_ids[k])
        resarr[k] = mleft != len(subcontents) and subcontents[mleft] == content_num_ids[k]
    return resarr

def has_correlations_test(content_num_ids, topic_num_ids):
    assert len(content_num_ids) == len(topic_num_ids)

    left = np.searchsorted(has_correlation_test_topics, topic_num_ids, side = "left")
    right = np.searchsorted(has_correlation_test_topics, topic_num_ids, side="right")

    resarr = np.zeros(shape = (len(content_num_ids)), dtype = bool)

    for k in range(len(content_num_ids)):
        subcontents = has_correlation_test_contents[left[k]:right[k]]
        mleft = np.searchsorted(subcontents, content_num_ids[k])
        resarr[k] = mleft != len(subcontents) and subcontents[mleft] == content_num_ids[k]
    return resarr

# -------------------------------- randomly sample from data --------------------------------
def obtain_train_sample(one_sample_size, zero_sample_size):
    one_samples = np.random.choice(len(has_correlation_train_topics), one_sample_size, replace = False)
    topics_num_id = has_correlation_train_topics[one_samples]
    contents_num_id = has_correlation_train_contents[one_samples]
    cor = np.ones(len(one_samples))

    zero_sample_topics = train_topics_num_id[np.random.choice(len(train_topics_num_id), zero_sample_size)]
    zero_sample_contents = train_contents_num_id[np.random.choice(len(train_contents_num_id), zero_sample_size)]
    zeroS_cor = has_correlations(zero_sample_contents, zero_sample_topics).astype(cor.dtype)

    topics_num_id = np.concatenate((topics_num_id, zero_sample_topics))
    contents_num_id = np.concatenate((contents_num_id, zero_sample_contents))
    cor = np.concatenate((cor, zeroS_cor))

    return topics_num_id, contents_num_id, cor

def obtain_test_sample(one_sample_size, zero_sample_size):
    one_samples = np.random.choice(len(has_correlation_test_topics), one_sample_size, replace=False)
    topics_num_id = has_correlation_test_topics[one_samples]
    contents_num_id = has_correlation_test_contents[one_samples]
    cor = np.ones(len(one_samples))

    zero_sample_topics = test_topics_num_id[np.random.choice(len(test_topics_num_id), zero_sample_size)]
    zero_sample_contents = test_contents_num_id[np.random.choice(len(test_contents_num_id), zero_sample_size)]
    zeroS_cor = has_correlations(zero_sample_contents, zero_sample_topics).astype(cor.dtype)

    topics_num_id = np.concatenate((topics_num_id, zero_sample_topics))
    contents_num_id = np.concatenate((contents_num_id, zero_sample_contents))
    cor = np.concatenate((cor, zeroS_cor))

    return topics_num_id, contents_num_id, cor
def obtain_train_square_sample(sample_size):
    topics = train_topics_num_id[np.random.choice(len(train_topics_num_id), sample_size, replace = False)]
    contents = train_contents_num_id[np.random.choice(len(train_contents_num_id), sample_size, replace=False)]

    topics = np.repeat(topics, sample_size)
    contents = np.tile(contents, sample_size)
    cor = has_correlations(contents, topics)

    return topics, contents, cor.astype(np.float64)
def obtain_test_square_sample(sample_size):
    topics = test_topics_num_id[np.random.choice(len(test_topics_num_id), sample_size, replace=False)]
    contents = test_contents_num_id[np.random.choice(len(test_contents_num_id), sample_size, replace=False)]

    topics = np.repeat(topics, sample_size)
    contents = np.tile(contents, sample_size)
    cor = has_correlations(contents, topics)

    return topics, contents, cor.astype(np.float64)

gc.collect()