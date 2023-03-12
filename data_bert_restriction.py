import gc

import numpy as np
import data_bert
import data_bert_tree_struct
import data_bert_sampler
import time

train_content_ids, train_topic_ids, test_content_ids, test_topic_ids = None, None, None, None

has_correlations_train_contents, has_correlations_train_topics = None, None
has_correlations_test_contents, has_correlations_test_topics = None, None

has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2 = None, None
has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2 = None, None

default_dampening_sampler_instance, default_dampening_sampler_overshoot2_instance = None, None

# adds the intersection of (contents_short_restriction, topics_short_restriction) and (contents_long_restriction, topics_long_restriction)
# into the list.
def add_potentials_into_list(contents_list, topics_list, contents_short_restriction, topics_short_restriction, contents_long_restriction, topics_long_restriction):
    has_content = data_bert.fast_contains_multi(data_bert.contents_availability_num_id, contents_short_restriction)
    contents_short_restriction = contents_short_restriction[has_content]
    topics_short_restriction = topics_short_restriction[has_content]
    del has_content
    has_content = data_bert.fast_contains_multi(data_bert.topics_availability_num_id, topics_short_restriction)
    contents_short_restriction = contents_short_restriction[has_content]
    topics_short_restriction = topics_short_restriction[has_content]

    contains = data_bert.has_correlations_general(contents_short_restriction, topics_short_restriction, contents_long_restriction, topics_long_restriction).astype(dtype=bool)
    contents_list.extend(list(contents_short_restriction[contains]))
    topics_list.extend(list(topics_short_restriction[contains]))
    del contains

def sort_according_to_topic(content_ids, topic_ids):
    sidx = np.argsort(topic_ids)
    sorted_topic = topic_ids[sidx]
    sorted_content = content_ids[sidx]
    del topic_ids, content_ids

    for topic_num_id in np.unique(sorted_topic):
        left = np.searchsorted(sorted_topic, topic_num_id, side="left")
        right = np.searchsorted(sorted_topic, topic_num_id, side="right")
        sorted_content[left:right] = np.sort(sorted_content[left:right])
    return sorted_content, sorted_topic

def generate_info_from_folder(combined_train_folder, combined_test_folder):
    global train_content_ids, train_topic_ids, test_content_ids, test_topic_ids
    global has_correlations_train_contents, has_correlations_train_topics
    global has_correlations_test_contents, has_correlations_test_topics
    global has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2
    global has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2

    train_file_lengths = np.load(combined_train_folder + "saved_files_lengths.npy")
    test_file_lengths = np.load(combined_test_folder + "saved_files_lengths.npy")

    train_content_ids = np.memmap("temp_train_contents.mmap", dtype=np.int32,
                                               shape=train_file_lengths.sum(), mode="w+")
    train_topic_ids = np.memmap("temp_train_topics.mmap", dtype=np.int32,
                                     shape=train_file_lengths.sum(), mode="w+")

    test_content_ids = np.memmap("temp_test_contents.mmap", dtype=np.int32,
                                       shape=test_file_lengths.sum(), mode="w+")
    test_topic_ids = np.memmap("temp_test_topics.mmap", dtype=np.int32,
                                     shape=test_file_lengths.sum(), mode="w+")

    has_correlations_train_contents, has_correlations_train_topics = [], []
    has_correlations_test_contents, has_correlations_test_topics = [], []

    has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2 = [], []
    has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2 = [], []

    rval = 0
    for k in range(len(train_file_lengths)):
        ctime = time.time()
        loaded_contents = np.load(combined_train_folder + str(k) + "_contents.npy")
        loaded_topics = np.load(combined_train_folder + str(k) + "_topics.npy")

        add_potentials_into_list(has_correlations_train_contents, has_correlations_train_topics, data_bert.has_correlation_contents,
                                 data_bert.has_correlation_topics, loaded_contents, loaded_topics)
        add_potentials_into_list(has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2, data_bert_tree_struct.has_close_correlation_contents,
                                 data_bert_tree_struct.has_close_correlation_topics, loaded_contents, loaded_topics)

        train_content_ids[rval:rval+train_file_lengths[k]] = loaded_contents
        train_topic_ids[rval:rval + train_file_lengths[k]] = loaded_topics
        rval += train_file_lengths[k]
        train_content_ids.flush()
        train_topic_ids.flush()

        del loaded_topics, loaded_contents
        ctime = time.time() - ctime
        print("Loaded ", k, "for train averages.   Time:", ctime)

    rval = 0
    for k in range(len(test_file_lengths)):
        ctime = time.time()
        loaded_contents = np.load(combined_test_folder + str(k) + "_contents.npy")
        loaded_topics = np.load(combined_test_folder + str(k) + "_topics.npy")

        add_potentials_into_list(has_correlations_test_contents, has_correlations_test_topics, data_bert.has_correlation_contents,
                                 data_bert.has_correlation_topics, loaded_contents, loaded_topics)
        add_potentials_into_list(has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2, data_bert_tree_struct.has_close_correlation_contents,
                                 data_bert_tree_struct.has_close_correlation_topics, loaded_contents, loaded_topics)

        test_content_ids[rval:rval + test_file_lengths[k]] = loaded_contents
        test_topic_ids[rval:rval + test_file_lengths[k]] = loaded_topics
        rval += test_file_lengths[k]
        test_content_ids.flush()
        test_topic_ids.flush()

        del loaded_topics, loaded_contents
        ctime = time.time() - ctime
        print("Loaded ", k, "for test averages.   Time:", ctime)

    has_correlations_train_contents = np.array(has_correlations_train_contents, dtype=np.int32)
    has_correlations_train_topics = np.array(has_correlations_train_topics, dtype=np.int32)
    has_correlations_test_contents = np.array(has_correlations_test_contents, dtype=np.int32)
    has_correlations_test_topics = np.array(has_correlations_test_topics, dtype=np.int32)

    has_correlations_train_contents_overshoot2 = np.array(has_correlations_train_contents_overshoot2, dtype=np.int32)
    has_correlations_train_topics_overshoot2 = np.array(has_correlations_train_topics_overshoot2, dtype=np.int32)
    has_correlations_test_contents_overshoot2 = np.array(has_correlations_test_contents_overshoot2, dtype=np.int32)
    has_correlations_test_topics_overshoot2 = np.array(has_correlations_test_topics_overshoot2, dtype=np.int32)

    has_correlations_train_contents, has_correlations_train_topics = sort_according_to_topic(has_correlations_train_contents, has_correlations_train_topics)
    has_correlations_test_contents, has_correlations_test_topics = sort_according_to_topic(has_correlations_test_contents, has_correlations_test_topics)
    has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2 = sort_according_to_topic(has_correlations_train_contents_overshoot2, has_correlations_train_topics_overshoot2)
    has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2 = sort_according_to_topic(has_correlations_test_contents_overshoot2, has_correlations_test_topics_overshoot2)

    gc.collect()

    global default_dampening_sampler_instance, default_dampening_sampler_overshoot2_instance
    default_dampening_sampler_instance = DampeningSampler(has_correlations_train_topics, has_correlations_train_contents,
                                                          has_correlations_test_topics, has_correlations_test_contents, data_bert.has_correlations)
    default_dampening_sampler_overshoot2_instance = DampeningSampler(has_correlations_train_topics_overshoot2, has_correlations_train_contents_overshoot2,
                                                          has_correlations_test_topics_overshoot2, has_correlations_test_contents_overshoot2, data_bert_tree_struct.has_close_correlations)

class DampeningSampler(data_bert_sampler.SamplerBase):
    def __init__(self, has_cor_train_topics, has_cor_train_contents, has_cor_test_topics, has_cor_test_contents, has_correlations_function):
        data_bert_sampler.SamplerBase.__init__(self)

        self.has_cor_train_topics = has_cor_train_topics
        self.has_cor_train_contents = has_cor_train_contents
        self.has_cor_test_topics = has_cor_test_topics
        self.has_cor_test_contents = has_cor_test_contents

        self.choice_gen = np.random.default_rng()
        self.has_correlations_function = has_correlations_function

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_sample(self, sample_size):
        has_cor_choice = self.choice_gen.choice(len(self.has_cor_train_topics), sample_size // 2, replace=False)
        topics = self.has_cor_train_topics[has_cor_choice]
        contents = self.has_cor_train_contents[has_cor_choice]
        cors = np.ones(len(topics))

        topics2, contents2, cors2, x = self.obtain_train_square_sample(sample_size // 2)
        topics = np.concatenate((topics, topics2))
        contents = np.concatenate((contents, contents2))
        cors = np.concatenate((cors, cors2))

        return topics, contents, cors, None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        has_cor_choice = self.choice_gen.choice(len(self.has_cor_test_topics), sample_size // 2, replace=False)
        topics = self.has_cor_test_topics[has_cor_choice]
        contents = self.has_cor_test_contents[has_cor_choice]
        cors = np.ones(len(topics))

        topics2, contents2, cors2, x = self.obtain_test_square_sample(sample_size // 2)
        topics = np.concatenate((topics, topics2))
        contents = np.concatenate((contents, contents2))
        cors = np.concatenate((cors, cors2))

        return topics, contents, cors, None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_square_sample(self, sample_size):
        """if global_epoch > self.prev_epoch:
            self.draw_new_train_test()
            self.prev_epoch = global_epoch """
        global train_content_ids, train_topic_ids
        choice = self.choice_gen.choice(len(train_content_ids), sample_size, replace=False)
        contents = train_content_ids[choice]
        topics = train_topic_ids[choice]
        return topics, contents, data_bert.has_correlations(contents, topics), None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        """if global_epoch > self.prev_epoch:
            self.draw_new_train_test()
            self.prev_epoch = global_epoch """
        global test_content_ids, test_topic_ids
        choice = self.choice_gen.choice(len(test_content_ids), sample_size, replace=False)
        contents = test_content_ids[choice]
        topics = test_topic_ids[choice]
        return topics, contents, data_bert.has_correlations(contents, topics), None

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        return self.has_correlations_function(content_num_ids, topic_num_ids)