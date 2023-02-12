# implements the same functions as for data_vectorizer, but with higher efficiency by directly returning
# indices and tensorflow tensors
import config
import data
import data_vectorizer
import tensorflow as tf
import numpy as np
import itertools
import pandas as pd
import os
import json

def transform_string_to_list(x):
    if type(x) == str:
        if len(x) == 2:
            return []
        return [int(n) for n in x[1:-1].split(", ")]
    if type(x) == list:
        return [int(n) for n in x]
    raise Exception("Unknown invalid type!")

contents_vect = data.contents[["title_vectorize", "description_vectorize"]]
topics_vect = data.topics[["title_vectorize", "description_vectorize"]]

contents_vect["title_vectorize"] = contents_vect["title_vectorize"].apply(transform_string_to_list)
contents_vect["description_vectorize"] = contents_vect["description_vectorize"].apply(transform_string_to_list)
topics_vect["title_vectorize"] = topics_vect["title_vectorize"].apply(transform_string_to_list)
topics_vect["description_vectorize"] = topics_vect["description_vectorize"].apply(transform_string_to_list)

train_contents_vect = contents_vect.loc[data_vectorizer.train_contents]
train_topics_vect = topics_vect.loc[data_vectorizer.train_topics]
test_contents_vect = contents_vect.loc[data_vectorizer.test_contents]
test_topics_vect = topics_vect.loc[data_vectorizer.test_topics]

train_contents_title_vect = tf.ragged.constant(list(train_contents_vect["title_vectorize"]))
train_contents_description_vect = tf.ragged.constant(list(train_contents_vect["description_vectorize"]))
train_topics_title_vect = tf.ragged.constant(list(train_topics_vect["title_vectorize"]))
train_topics_description_vect = tf.ragged.constant(list(train_topics_vect["description_vectorize"]))
test_contents_title_vect = tf.ragged.constant(list(test_contents_vect["title_vectorize"]))
test_contents_description_vect = tf.ragged.constant(list(test_contents_vect["description_vectorize"]))
test_topics_title_vect = tf.ragged.constant(list(test_topics_vect["title_vectorize"]))
test_topics_description_vect = tf.ragged.constant(list(test_topics_vect["description_vectorize"]))
def obtain_correlations(contents_index, topics_index):
    contain_contents_list = []
    lookup_table = pd.Series(index=contents_index, data=range(len(contents_index)))
    for k in range(len(topics_index)):
        if k % 100 == 0:
            print("Load idx:  ",k)
        topic_id = topics_index[k]
        if data.topics.loc[topic_id]["has_content"]:
            content_ids = pd.Index(data.correlations.loc[topic_id]["content_ids"].split())
            content_ids = list(lookup_table.loc[content_ids.intersection(contents_index)])
            contain_contents_list.append(content_ids)
        else:
            contain_contents_list.append([])
    return list(contain_contents_list)

if os.path.isfile(config.resources_path + "vectorizer_train_correlations.json"):
    with open(config.resources_path + "vectorizer_train_correlations.json") as f:
        train_correlations = json.load(f)
else:
    print("First run train correlations: ")
    train_correlations = obtain_correlations(data_vectorizer.train_contents, data_vectorizer.train_topics)
    with open(config.resources_path + 'vectorizer_train_correlations.json', 'w') as f:
        json.dump(train_correlations, f)
if os.path.isfile(config.resources_path + "vectorizer_test_correlations.json"):
    with open(config.resources_path + "vectorizer_test_correlations.json") as f:
        test_correlations = json.load(f)
else:
    print("First run test correlations: ")
    test_correlations = obtain_correlations(data_vectorizer.test_contents, data_vectorizer.test_topics)
    with open(config.resources_path + 'vectorizer_test_correlations.json', 'w') as f:
        json.dump(test_correlations, f)

train_correlations_length = np.zeros(shape = (len(train_correlations)), dtype = np.int32)
test_correlations_length = np.zeros(shape = (len(test_correlations)), dtype = np.int32)
for k in range(len(train_correlations)):
    train_correlations[k] = np.array(train_correlations[k], dtype = np.int32)
    train_correlations_length[k] = len(train_correlations[k])
train_correlations = np.array(train_correlations, dtype = "object")
for k in range(len(test_correlations)):
    test_correlations[k] = np.array(test_correlations[k], dtype = np.int32)
    test_correlations_length[k] = len(test_correlations[k])
test_correlations = np.array(test_correlations, dtype = "object")

# asssumes the contents list and topics list are sorted.
def obtain_correlations_matrix_train(topics_list_train_id, contents_list_train_id):
    cors = np.zeros(shape = (len(topics_list_train_id), len(contents_list_train_id)))
    inv_map = np.zeros(shape = (len(data_vectorizer.train_contents)), dtype = np.int32) - 1
    inv_map[contents_list_train_id] = np.arange(0, len(contents_list_train_id), dtype = np.int32)
    # np.concatenate(np.train_correlations[topics_list_train_id], axis = 0)
    # np.repeat(np.array(range(len(topics_list_train_id))), train_correlations_length[topics_list_train_id])
    k = 0
    for topic_id in topics_list_train_id:
        locs = inv_map[train_correlations[topic_id]]
        cors[k, locs[locs != -1]] = 1
        k += 1
    return cors

def obtain_correlations_matrix_test(topics_list_test_id, contents_list_test_id):
    cors = np.zeros(shape=(len(topics_list_test_id), len(contents_list_test_id)))
    inv_map = np.zeros(shape=(len(data_vectorizer.test_contents)), dtype=np.int32) - 1
    inv_map[contents_list_test_id] = np.arange(0, len(contents_list_test_id), dtype=np.int32)

    k = 0
    for topic_id in topics_list_test_id:
        locs = inv_map[test_correlations[topic_id]]
        cors[k, locs[locs != -1]] = 1
        k += 1
    return cors


def obtain_train_contents_vector(contents_list_train_id):
    return tf.gather(train_contents_title_vect, contents_list_train_id), tf.gather(train_contents_description_vect, contents_list_train_id)

def obtain_train_topics_vector(topics_list_train_id):
    return tf.gather(train_topics_title_vect, topics_list_train_id), tf.gather(train_topics_description_vect, topics_list_train_id)

def obtain_test_contents_vector(contents_list_train_id):
    return tf.gather(test_contents_title_vect, contents_list_train_id), tf.gather(test_contents_description_vect, contents_list_train_id)

def obtain_test_topics_vector(topics_list_train_id):
    return tf.gather(test_topics_title_vect, topics_list_train_id), tf.gather(test_topics_description_vect, topics_list_train_id)


# same as that of data_vectorizer, except it returns directly the tf ragged tensors.
def random_train_batch_sample(initial_sample_size = 1000, zero_to_one_ratio = None):
    contents_list = np.random.choice(len(data_vectorizer.train_contents), initial_sample_size, replace=False)
    topics_list = np.random.choice(len(data_vectorizer.train_topics), initial_sample_size, replace=False)
    cor_mat = obtain_correlations_matrix_train(topics_list, contents_list)

    if zero_to_one_ratio is None:
        return np.tile(contents_list, len(topics_list)), np.repeat(topics_list, len(contents_list)), cor_mat.flatten()

    if initial_sample_size < 800:
        raise Exception("Zero to one ratio can be used only if initial sample size >= 1000!")

    while cor_mat.sum() < 10:
        contents_list = np.random.choice(len(data_vectorizer.train_contents), initial_sample_size, replace=False)
        topics_list = np.random.choice(len(data_vectorizer.train_topics), initial_sample_size, replace=False)
        cor_mat = obtain_correlations_matrix_train(topics_list, contents_list)

    one_locations = np.where(cor_mat == 1)
    zero_locations = np.where(cor_mat == 0)
    stopics = one_locations[0]
    scontents = one_locations[1]

    one_length = len(stopics)
    zero_length = int(one_length * zero_to_one_ratio)
    correlations = [1] * one_length
    correlations.extend([0] * zero_length)

    zero_samples = list(np.random.choice(len(zero_locations[0]), zero_length, replace=False))
    stopics = np.concatenate([stopics, zero_locations[0][zero_samples]])
    scontents = np.concatenate([scontents, zero_locations[1][zero_samples]])

    stopics = topics_list[stopics]
    scontents = contents_list[scontents]

    return scontents, stopics, correlations

def random_test_batch_sample(initial_sample_size = 1000, zero_to_one_ratio = None):
    contents_list = np.random.choice(len(data_vectorizer.test_contents), initial_sample_size, replace=False)
    topics_list = np.random.choice(len(data_vectorizer.test_topics), initial_sample_size, replace=False)
    cor_mat = obtain_correlations_matrix_test(topics_list, contents_list)

    if zero_to_one_ratio is None:
        return np.tile(contents_list, len(topics_list)), np.repeat(topics_list, len(contents_list)), cor_mat.flatten()

    if initial_sample_size < 800:
        raise Exception("Zero to one ratio can be used only if initial sample size >= 1000!")

    while cor_mat.sum() < 10:
        contents_list = np.random.choice(len(data_vectorizer.test_contents), initial_sample_size, replace=False)
        topics_list = np.random.choice(len(data_vectorizer.test_topics), initial_sample_size, replace=False)
        cor_mat = obtain_correlations_matrix_test(topics_list, contents_list)

    one_locations = np.where(cor_mat == 1)
    zero_locations = np.where(cor_mat == 0)
    stopics = one_locations[0]
    scontents = one_locations[1]

    one_length = len(stopics)
    zero_length = int(one_length * zero_to_one_ratio)
    correlations = [1] * one_length
    correlations.extend([0] * zero_length)

    zero_samples = list(np.random.choice(len(zero_locations[0]), zero_length, replace=False))
    stopics = np.concatenate([stopics, zero_locations[0][zero_samples]])
    scontents = np.concatenate([scontents, zero_locations[1][zero_samples]])

    stopics = topics_list[stopics]
    scontents = contents_list[scontents]

    return scontents, stopics, correlations
