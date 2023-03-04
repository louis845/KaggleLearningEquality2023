import pandas as pd
import numpy as np
import json

# basically a stripped down version of data_bert.

contents = pd.read_csv("../data/" + "contents_translate.csv", index_col = 0)
correlations = pd.read_csv("../data/" + "correlations.csv", index_col = 0)
topics = pd.read_csv("../data/" + "topics_translate.csv", index_col = 0)

# inverse map, to obtain the integer id of topic / content given the str id.
contents_inv_map = pd.Series(data = np.arange(len(contents)), index = contents.index)
topics_inv_map = pd.Series(data = np.arange(len(topics)), index = topics.index)

with open("../data/" + "temp_channel_components.json") as file:
    channel_components = json.loads(file.read())
with open("../data/" + "temp_topic_trees_contents.json") as file:
    topic_trees_contents = json.loads(file.read())

# -------------------------------- restrict the topics and contents to those that have at least 1 description --------------------------------
contents_availability = np.zeros(shape = (len(contents), 3), dtype = bool)
contents_availability[:, 0] = np.load("../data/" + "bert_tokens/contents_description/has_content_mask.npy")
contents_availability[:, 1] = np.load("../data/" + "bert_tokens/contents_title/has_content_mask.npy")
contents_availability[:, 2] = np.logical_or(contents_availability[:, 0], contents_availability[:, 1])
contents_availability_num_id = np.where(np.logical_or(contents_availability[:, 0], contents_availability[:, 1]))[0]
contents_availability = pd.DataFrame(data = contents_availability, columns = ["description", "title", "some_available"], index = contents.index)

topics_availability = np.zeros(shape = (len(topics), 3), dtype = bool)
topics_availability[:, 0] = np.load("../data/" + "bert_tokens/topics_description/has_content_mask.npy")
topics_availability[:, 1] = np.load("../data/" + "bert_tokens/topics_title/has_content_mask.npy")
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

# see if sorted_arr contains val, returns a bool
def fast_contains(sorted_arr, val):
    assert (sorted_arr[1:] < sorted_arr[:-1]).sum() == 0
    idx = np.searchsorted(sorted_arr, val, side = "left")
    return idx != len(sorted_arr) and sorted_arr[idx] == val

# see if sorted_arr contains vals, returns a bool array
def fast_contains_multi(sorted_arr, vals):
    assert (sorted_arr[1:] < sorted_arr[:-1]).sum() == 0
    return np.searchsorted(sorted_arr, vals, side = "right") > np.searchsorted(sorted_arr, vals, side = "left")