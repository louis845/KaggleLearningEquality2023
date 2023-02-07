# implements the same functions as for data_vectorizer, but with higher efficiency by directly returning
# indices and tensorflow tensors

import data
import data_vectorizer
import tensorflow as tf
import numpy as np
import itertools

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
    cor_frame = data_vectorizer.obtain_correlation_frame(list(data_vectorizer.train_topics[topics_list]), list(data_vectorizer.train_contents[contents_list]))

    if zero_to_one_ratio is None:
        prod = itertools.product(list(topics_list), list(contents_list))
        lists = list(zip(*prod))
        return list(lists[1]), list(lists[0]), cor_frame.to_numpy().flatten()

    if initial_sample_size < 1000:
        raise Exception("Zero to one ratio can be used only if initial sample size >= 1000!")

    while cor_frame.to_numpy().sum() < 10:
        contents_list = np.random.choice(len(data_vectorizer.train_contents), initial_sample_size, replace=False)
        topics_list = np.random.choice(len(data_vectorizer.train_topics), initial_sample_size, replace=False)
        cor_frame = data_vectorizer.obtain_correlation_frame(list(data_vectorizer.train_topics[topics_list]), list(data_vectorizer.train_contents[contents_list]))

    cor_mat = cor_frame.to_numpy().astype(dtype=np.int32)

    one_locations = np.where(cor_mat == 1)
    zero_locations = np.where(cor_mat == 0)
    stopics = list(one_locations[0])
    scontents = list(one_locations[1])

    one_length = len(stopics)
    zero_length = int(one_length * zero_to_one_ratio)
    correlations = [1] * one_length
    correlations.extend([0] * zero_length)

    zero_samples = list(np.random.choice(len(zero_locations[0]), zero_length, replace=False))
    stopics.extend(zero_locations[0][zero_samples])
    scontents.extend(zero_locations[1][zero_samples])

    stopics = topics_list[stopics]
    scontents = contents_list[scontents]

    return scontents, stopics, correlations
