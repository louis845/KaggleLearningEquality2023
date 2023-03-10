import config
import numpy as np
import pandas as pd
import json
import gc
import os

contents = pd.read_csv(config.resources_path + "contents_translate.csv", index_col = 0)
correlations = pd.read_csv(config.resources_path + "correlations.csv", index_col = 0)
topics = pd.read_csv(config.resources_path + "topics_translate.csv", index_col = 0)

# inverse map, to obtain the integer id of topic / content given the str id.
contents_inv_map = pd.Series(data = np.arange(len(contents)), index = contents.index)
topics_inv_map = pd.Series(data = np.arange(len(topics)), index = topics.index)

with open(config.resources_path + "temp_channel_components.json") as file:
    channel_components = json.loads(file.read())
with open(config.resources_path + "temp_topic_trees_contents.json") as file:
    topic_trees_contents = json.loads(file.read())

# -------------------------------- restrict the topics and contents to those that have at least 1 description --------------------------------
contents_availability = np.zeros(shape = (len(contents), 3), dtype = bool)
contents_availability[:, 0] = np.load(config.resources_path + "bert_tokens/contents_description/has_content_mask.npy")
contents_availability[:, 1] = np.load(config.resources_path + "bert_tokens/contents_title/has_content_mask.npy")
contents_availability[:, 2] = np.logical_or(contents_availability[:, 0], contents_availability[:, 1])
contents_availability_num_id = np.where(np.logical_or(contents_availability[:, 0], contents_availability[:, 1]))[0]
contents_availability = pd.DataFrame(data = contents_availability, columns = ["description", "title", "some_available"], index = contents.index)

topics_availability = np.zeros(shape = (len(topics), 3), dtype = bool)
topics_availability[:, 0] = np.load(config.resources_path + "bert_tokens/topics_description/has_content_mask.npy")
topics_availability[:, 1] = np.load(config.resources_path + "bert_tokens/topics_title/has_content_mask.npy")
topics_availability[:, 2] = np.logical_or(topics_availability[:, 0], topics_availability[:, 1])
topics_availability_num_id = np.where(np.logical_or(topics_availability[:, 0], topics_availability[:, 1]))[0]
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
train_contents_num_id = np.unique(contents_inv_map[train_contents].to_numpy())
train_topics_num_id = np.unique(topics_inv_map[train_topics].to_numpy())
test_contents_num_id = np.unique(contents_inv_map[test_contents].to_numpy())
test_topics_num_id = np.unique(topics_inv_map[test_topics].to_numpy())

train_contents = contents.index[train_contents_num_id]
train_topics = topics.index[train_topics_num_id]
test_contents = contents.index[test_contents_num_id]
test_topics = topics.index[test_topics_num_id]

# -------------------------------- create association matrix from correlations --------------------------------
# a list of tuples (topic, content) such that there exists a correlation between them
# see if sorted_arr contains val
def fast_contains(sorted_arr, val):
    idx = np.searchsorted(sorted_arr, val, side = "left")
    return idx != len(sorted_arr) and sorted_arr[idx] == val

# see if sorted_arr contains vals, returns a bool array
def fast_contains_multi(sorted_arr, vals):
    assert (sorted_arr[1:] < sorted_arr[:-1]).sum() == 0
    return np.searchsorted(sorted_arr, vals, side = "right") > np.searchsorted(sorted_arr, vals, side = "left")

if os.path.isdir(config.resources_path + "data_bert/"):
    has_correlation_topics = np.load(config.resources_path + "data_bert/has_correlation_topics.npy")
    has_correlation_contents = np.load(config.resources_path + "data_bert/has_correlation_contents.npy")
    has_correlation_train_topics = np.load(config.resources_path + "data_bert/has_correlation_train_topics.npy")
    has_correlation_train_contents = np.load(config.resources_path + "data_bert/has_correlation_train_contents.npy")
    has_correlation_test_topics = np.load(config.resources_path + "data_bert/has_correlation_test_topics.npy")
    has_correlation_test_contents = np.load(config.resources_path + "data_bert/has_correlation_test_contents.npy")
else:
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
                to_add = np.array(contents_inv_map[content_ids], dtype = np.int32)
                mask = fast_contains_multi(train_contents_num_id, to_add)
                to_add = np.sort(to_add[mask])
                has_correlation_train_topics.extend([k] * len(to_add))
                has_correlation_train_contents.extend(list(to_add))


            if k in test_topics_num_id:
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

    os.mkdir(config.resources_path + "data_bert/")
    np.save(config.resources_path + "data_bert/has_correlation_topics.npy", has_correlation_topics)
    np.save(config.resources_path + "data_bert/has_correlation_contents.npy", has_correlation_contents)
    np.save(config.resources_path + "data_bert/has_correlation_train_topics.npy", has_correlation_train_topics)
    np.save(config.resources_path + "data_bert/has_correlation_train_contents.npy", has_correlation_train_contents)
    np.save(config.resources_path + "data_bert/has_correlation_test_topics.npy", has_correlation_test_topics)
    np.save(config.resources_path + "data_bert/has_correlation_test_contents.npy", has_correlation_test_contents)

def has_correlation(content_num_id, topic_num_id):
    left = np.searchsorted(has_correlation_topics, topic_num_id, side = "left")
    right = np.searchsorted(has_correlation_topics, topic_num_id, side = "right")

    if left == len(has_correlation_topics) or has_correlation_topics[left] != topic_num_id:
        return False

    subcontents = has_correlation_contents[left:right]
    left = np.searchsorted(subcontents, content_num_id)
    return left != len(subcontents) and subcontents[left] == content_num_id

# same function as above, except we accept a list of points, and the arrays which represent the topics and contents corrs.
def has_correlations_general(content_num_ids, topic_num_ids, corr_contents_arr, corr_topics_arr):
    assert len(content_num_ids) == len(topic_num_ids)

    left = np.searchsorted(corr_topics_arr, topic_num_ids, side = "left")
    right = np.searchsorted(corr_topics_arr, topic_num_ids, side="right")

    resarr = np.zeros(shape = (len(content_num_ids)), dtype = bool)

    for k in range(len(content_num_ids)):
        subcontents = corr_contents_arr[left[k]:right[k]]
        mleft = np.searchsorted(subcontents, content_num_ids[k])
        resarr[k] = mleft != len(subcontents) and subcontents[mleft] == content_num_ids[k]
    return resarr
def has_correlations(content_num_ids, topic_num_ids):
    return has_correlations_general(content_num_ids, topic_num_ids, has_correlation_contents, has_correlation_topics)

def has_correlations_train(content_num_ids, topic_num_ids):
    return has_correlations_general(content_num_ids, topic_num_ids, has_correlation_train_contents, has_correlation_train_topics)

def has_correlations_test(content_num_ids, topic_num_ids):
    return has_correlations_general(content_num_ids, topic_num_ids, has_correlation_test_contents, has_correlation_test_topics)

rng_seed = np.random.default_rng()

# -------------------------------- randomly sample from data --------------------------------
# one_sample_size and zero_sample_size are the sizes of sample of ones or sample of zeros. note that the zeros may actually
# be ones since they are randomly drawn with replacement from the whole topics_restricted and contents_restricted.
# corr_contents_arr and corr_topics_arr are the tuples representing whether there is a correlation between the contents and topics.
# topics_restricted, contents_restricted are all the topics and contents we have to draw from.
# IMPORTANT: note that corr_contents_arr and corr_topics_arr have to ONLY contain contents and topics that are from topics_restricted
# and contents_restricted.
def obtain_general_sample(one_sample_size, zero_sample_size, corr_contents_arr, corr_topics_arr, contents_restricted, topics_restricted):
    one_samples = rng_seed.choice(len(corr_topics_arr), one_sample_size, replace = False)
    topics_num_id = corr_topics_arr[one_samples]
    contents_num_id = corr_contents_arr[one_samples]
    cor = np.ones(len(one_samples))

    zero_sample_topics = topics_restricted[rng_seed.choice(len(topics_restricted), zero_sample_size)]
    zero_sample_contents = contents_restricted[rng_seed.choice(len(contents_restricted), zero_sample_size)]
    zeroS_cor = has_correlations_general(zero_sample_contents, zero_sample_topics, corr_contents_arr, corr_topics_arr).astype(cor.dtype)

    topics_num_id = np.concatenate((topics_num_id, zero_sample_topics))
    contents_num_id = np.concatenate((contents_num_id, zero_sample_contents))
    cor = np.concatenate((cor, zeroS_cor))

    return topics_num_id, contents_num_id, cor

def obtain_general_square_sample(sample_size, corr_contents_arr, corr_topics_arr, contents_restricted, topics_restricted):
    topics = topics_restricted[rng_seed.choice(len(topics_restricted), sample_size, replace = False)]
    contents = contents_restricted[rng_seed.choice(len(contents_restricted), sample_size, replace=False)]

    topics = np.repeat(topics, sample_size)
    contents = np.tile(contents, sample_size)
    cor = has_correlations_general(contents, topics, corr_contents_arr, corr_topics_arr)

    return topics, contents, cor.astype(np.float64)

def obtain_train_sample(one_sample_size, zero_sample_size):
    return obtain_general_sample(one_sample_size, zero_sample_size, has_correlation_train_contents, has_correlation_train_topics, train_contents_num_id, train_topics_num_id)

def obtain_test_sample(one_sample_size, zero_sample_size):
    return obtain_general_sample(one_sample_size, zero_sample_size, has_correlation_test_contents, has_correlation_test_topics, test_contents_num_id, test_topics_num_id)
def obtain_train_square_sample(sample_size):
    return obtain_general_square_sample(sample_size, has_correlation_train_contents, has_correlation_train_topics, train_contents_num_id, train_topics_num_id)
def obtain_test_square_sample(sample_size):
    return obtain_general_square_sample(sample_size, has_correlation_test_contents, has_correlation_test_topics, test_contents_num_id, test_topics_num_id)

gc.collect()