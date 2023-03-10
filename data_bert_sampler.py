# class to manage sampling and correlation verification. this class is the most general sampling from data method, which
# supports sampling from overshoot samples, tree samples and so on. also, it supports dividing the sampled parts into
# disjoint subsets, where the samples from each subset are drawn from different respective overshoot samples.
import gc
import math
import time

import numpy as np
import data_bert
import data_bert_tree_struct

import data_bert_hacky_disable_tree_struct

global_epoch = 0

class SamplerBase:

    def __init__(self):
        pass

    # this sampler includes class_ids and classes as arguments, to indicate which class that sample belongs to.
    # each subclass that extends SamplerBase might have different class IDs for the same class. Note that the results from
    # samplerbase are expected to be plugged back into the same sampler base.

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_sample(self, sample_size):
        pass

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        pass

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_square_sample(self, sample_size):
        pass

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        pass

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        pass

    def is_tree_sampler(self):
        return False

class DefaultSampler(SamplerBase):
    def __init__(self, sample_generation_functions = None, sample_verification_function = None):
        SamplerBase.__init__(self)
        assert (sample_generation_functions is None) == (sample_verification_function == None)
        if sample_generation_functions is None:
            self.sample_generation_functions = {
                "train_sample": data_bert.obtain_train_sample,
                "test_sample": data_bert.obtain_test_sample,
                "train_square_sample": data_bert.obtain_train_square_sample,
                "test_square_sample": data_bert.obtain_test_square_sample
            }  # a dict "train_sample", "test_sample", "train_square_sample", "test_square_sample"  from which we can draw samples from.

            self.sample_verification_function = data_bert.has_correlations  # the function to verify whether a tuple (content, topic) has correlations.
        else:
            self.sample_generation_functions = sample_generation_functions
            self.sample_verification_function = sample_verification_function

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_sample(self, sample_size):
        topics, contents, cors = self.sample_generation_functions["train_sample"](
            one_sample_size=sample_size // 2,
            zero_sample_size=sample_size // 2)
        return topics, contents, cors, None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        topics, contents, cors = self.sample_generation_functions["test_sample"](
            one_sample_size=sample_size // 2,
            zero_sample_size=sample_size // 2)
        return topics, contents, cors, None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_square_sample(self, sample_size):
        topics, contents, cors = self.sample_generation_functions["train_square_sample"](int(math.sqrt(sample_size)))
        return topics, contents, cors, None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        topics, contents, cors = self.sample_generation_functions["test_square_sample"](int(math.sqrt(sample_size)))
        return topics, contents, cors, None

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        return self.sample_verification_function(content_num_ids, topic_num_ids)

class DampeningSampler(SamplerBase):
    def __init__(self, combined_train_folder, combined_test_folder, topics_cor_file, contents_cor_file):
        SamplerBase.__init__(self)
        has_cor_topics = np.load(topics_cor_file)
        has_cor_contents = np.load(contents_cor_file)

        train_topics_pos = data_bert.fast_contains_multi(data_bert.train_topics_num_id, has_cor_topics)
        train_contents_pos = data_bert.fast_contains_multi(data_bert.train_contents_num_id, has_cor_contents)
        test_topics_pos = data_bert.fast_contains_multi(data_bert.test_topics_num_id, has_cor_topics)
        test_contents_pos = data_bert.fast_contains_multi(data_bert.test_contents_num_id, has_cor_contents)

        self.has_cor_train_topics = has_cor_topics[np.logical_and(train_topics_pos, train_contents_pos)]
        self.has_cor_train_contents = has_cor_contents[np.logical_and(train_topics_pos, train_contents_pos)]
        self.has_cor_test_topics = has_cor_topics[np.logical_and(test_topics_pos, test_contents_pos)]
        self.has_cor_test_contents = has_cor_contents[np.logical_and(test_topics_pos, test_contents_pos)]

        del has_cor_topics, has_cor_contents, train_topics_pos, train_contents_pos, test_topics_pos, test_contents_pos

        self.combined_train_folder = combined_train_folder
        self.combined_test_folder = combined_test_folder

        self.train_file_lengths = np.load(self.combined_train_folder + "saved_files_lengths.npy")
        self.test_file_lengths = np.load(self.combined_test_folder + "saved_files_lengths.npy")

        self.train_averages = self.train_file_lengths.astype(dtype=np.float64) / self.train_file_lengths.sum()
        self.test_averages = self.test_file_lengths.astype(dtype=np.float64) / self.test_file_lengths.sum()

        self.train_content_ids = np.memmap("temp_train_contents.mmap", dtype=np.int32,
                                           shape=self.train_file_lengths.sum(), mode="r")
        self.train_topic_ids = np.memmap("temp_train_topics.mmap", dtype=np.int32,
                                         shape=self.train_file_lengths.sum(), mode="r")

        self.test_content_ids = np.memmap("temp_test_contents.mmap", dtype=np.int32,
                                           shape=self.test_file_lengths.sum(), mode="r")
        self.test_topic_ids = np.memmap("temp_test_topics.mmap", dtype=np.int32,
                                         shape=self.test_file_lengths.sum(), mode="r")

        rval = 0
        for k in range(len(self.train_averages)):
            ctime = time.time()
            self.train_content_ids[rval:rval+self.train_file_lengths[k]] = np.load(self.combined_train_folder + str(k) + "_contents.npy")
            self.train_topic_ids[rval:rval + self.train_file_lengths[k]] = np.load(self.combined_train_folder + str(k) + "_topics.npy")
            rval += self.train_file_lengths[k]
            self.train_content_ids.flush()
            self.train_topic_ids.flush()
            ctime = time.time() - ctime
            print("Loaded ", k, "for train averages.   Time:", ctime)

        rval = 0
        for k in range(len(self.test_averages)):
            ctime = time.time()
            self.test_content_ids[rval:rval + self.test_file_lengths[k]] = np.load(
                self.combined_test_folder + str(k) + "_contents.npy")
            self.test_topic_ids[rval:rval + self.test_file_lengths[k]] = np.load(
                self.combined_test_folder + str(k) + "_topics.npy")
            rval += self.test_file_lengths[k]
            self.test_content_ids.flush()
            self.test_topic_ids.flush()
            ctime = time.time() - ctime
            print("Loaded ", k, "for train averages.   Time:", ctime)

        self.prev_epoch = 0

        self.choice_gen = np.random.default_rng()

    def draw_new_train_test(self):
        del self.train_content_ids, self.train_topic_ids, self.test_content_ids, self.test_topic_ids

        k1 = np.random.choice(len(self.train_averages), p=self.train_averages)
        self.train_content_ids = np.load(self.combined_train_folder + str(k1) + "_contents.npy")
        self.train_topic_ids = np.load(self.combined_train_folder + str(k1) + "_topics.npy")

        k2 = np.random.choice(len(self.test_averages), p=self.test_averages)
        self.test_content_ids = np.load(self.combined_test_folder + str(k2) + "_contents.npy")
        self.test_topic_ids = np.load(self.combined_test_folder + str(k2) + "_topics.npy")

        print("Loading new train set................", k1, "    ", k2)

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
        choice = self.choice_gen.choice(len(self.train_content_ids), sample_size, replace=False)
        contents = self.train_content_ids[choice]
        topics = self.train_topic_ids[choice]
        return topics, contents, data_bert.has_correlations(contents, topics), None

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        """if global_epoch > self.prev_epoch:
            self.draw_new_train_test()
            self.prev_epoch = global_epoch """
        choice = self.choice_gen.choice(len(self.test_content_ids), sample_size, replace=False)
        contents = self.test_content_ids[choice]
        topics = self.test_topic_ids[choice]
        return topics, contents, data_bert.has_correlations(contents, topics), None

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        return data_bert.has_correlations(content_num_ids, topic_num_ids)

class ReversedSampler(SamplerBase):
    def __init__(self, sampler):
        SamplerBase.__init__(self)
        assert not sampler.is_tree_sampler()
        self.sampler = sampler

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_sample(self, sample_size):
        return self.sampler.obtain_test_sample(sample_size)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        return self.sampler.obtain_train_sample(sample_size)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_square_sample(self, sample_size):
        return self.sampler.obtain_test_square_sample(sample_size)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        return self.sampler.obtain_train_square_sample(sample_size)

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        return self.sampler.has_correlations(content_num_ids, topic_num_ids, class_ids)

def obtain_tree_train_sample0(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_train_sample(one_sample_size, zero_sample_size, 0)

def obtain_tree_test_sample0(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_test_sample(one_sample_size, zero_sample_size, 0)

def obtain_tree_train_square_sample0(sample_size):
    return data_bert_tree_struct.obtain_tree_train_square_sample(sample_size, 0)

def obtain_tree_test_square_sample0(sample_size):
    return data_bert_tree_struct.obtain_tree_test_square_sample(sample_size, 0)

def obtain_tree_train_sample1(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_train_sample(one_sample_size, zero_sample_size, 1)

def obtain_tree_test_sample1(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_test_sample(one_sample_size, zero_sample_size, 1)

def obtain_tree_train_square_sample1(sample_size):
    return data_bert_tree_struct.obtain_tree_train_square_sample(sample_size, 1)

def obtain_tree_test_square_sample1(sample_size):
    return data_bert_tree_struct.obtain_tree_test_square_sample(sample_size, 1)

def obtain_tree_train_sample2(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_train_sample(one_sample_size, zero_sample_size, 2)

def obtain_tree_test_sample2(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_test_sample(one_sample_size, zero_sample_size, 2)

def obtain_tree_train_square_sample2(sample_size):
    return data_bert_tree_struct.obtain_tree_train_square_sample(sample_size, 2)

def obtain_tree_test_square_sample2(sample_size):
    return data_bert_tree_struct.obtain_tree_test_square_sample(sample_size, 2)

def obtain_tree_train_sample3(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_train_sample(one_sample_size, zero_sample_size, 3)

def obtain_tree_test_sample3(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_test_sample(one_sample_size, zero_sample_size, 3)

def obtain_tree_train_square_sample3(sample_size):
    return data_bert_tree_struct.obtain_tree_train_square_sample(sample_size, 3)

def obtain_tree_test_square_sample3(sample_size):
    return data_bert_tree_struct.obtain_tree_test_square_sample(sample_size, 3)

def obtain_tree_train_sample4(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_train_sample(one_sample_size, zero_sample_size, 4)

def obtain_tree_test_sample4(one_sample_size, zero_sample_size):
    return data_bert_tree_struct.obtain_tree_test_sample(one_sample_size, zero_sample_size, 4)

def obtain_tree_train_square_sample4(sample_size):
    return data_bert_tree_struct.obtain_tree_train_square_sample(sample_size, 4)

def obtain_tree_test_square_sample4(sample_size):
    return data_bert_tree_struct.obtain_tree_test_square_sample(sample_size, 4)

def has_tree_correlations0(content_num_ids, topic_lvk_num_ids):
    return data_bert_tree_struct.has_tree_correlations(content_num_ids, topic_lvk_num_ids, 0)

def has_tree_correlations1(content_num_ids, topic_lvk_num_ids):
    return data_bert_tree_struct.has_tree_correlations(content_num_ids, topic_lvk_num_ids, 1)

def has_tree_correlations2(content_num_ids, topic_lvk_num_ids):
    return data_bert_tree_struct.has_tree_correlations(content_num_ids, topic_lvk_num_ids, 2)

def has_tree_correlations3(content_num_ids, topic_lvk_num_ids):
    return data_bert_tree_struct.has_tree_correlations(content_num_ids, topic_lvk_num_ids, 3)

def has_tree_correlations4(content_num_ids, topic_lvk_num_ids):
    return data_bert_tree_struct.has_tree_correlations(content_num_ids, topic_lvk_num_ids, 4)

train_sample_pack = [obtain_tree_train_sample0, obtain_tree_train_sample1, obtain_tree_train_sample2,
                     obtain_tree_train_sample3, obtain_tree_train_sample4]

test_sample_pack = [obtain_tree_test_sample0, obtain_tree_test_sample1, obtain_tree_test_sample2,
                     obtain_tree_test_sample3, obtain_tree_test_sample4]

train_square_sample_pack = [obtain_tree_train_square_sample0, obtain_tree_train_square_sample1, obtain_tree_train_square_sample2,
                     obtain_tree_train_square_sample3, obtain_tree_train_square_sample4]

test_square_sample_pack = [obtain_tree_test_square_sample0, obtain_tree_test_square_sample1, obtain_tree_test_square_sample2,
                     obtain_tree_test_square_sample3, obtain_tree_test_square_sample4]

has_tree_correlations_pack = [has_tree_correlations0, has_tree_correlations1, has_tree_correlations2,
                              has_tree_correlations3, has_tree_correlations4]

class DefaultTreeSampler(SamplerBase):
    # the tree generation and tree verification functions are lists with the same length.
    # sample_tree_generation_functions[k] contain a dict for generating train samples, test samples,
    # train square sample, test square samples at the kth level. sample_tree_verification_functions[k]
    # is the function for verifying at the kth level.

    # the last level of the list is always the individual level. this means sampling normally.

    # the generation_sizes specify how many (square) samples to draw from, when sampling from training data.
    # this means a generation_sizes[k] x generation_sizes[k] region of topic_klevel_id and contents_id will
    # be drawn from training.

    # the last level (individual level) is sampled normally (with one size == zero size), so we do not need
    # generation_sizes[-1]. this means len(generation_sizes) == len(sample_tree_generation_functions)-1
    # similarly, len(sample_tree_abundances) == len(sample_tree_generation_functions)-1

    # the custom sample generation functions and classes are not fully supported now
    def __init__(self, sample_tree_generation_functions = None, sample_tree_verification_functions = None,
                 sample_tree_abundances_train = None, sample_tree_abundances_test = None, generation_sizes = None):
        SamplerBase.__init__(self)
        assert (sample_tree_generation_functions is None) == (sample_tree_verification_functions == None)
        assert (sample_tree_generation_functions is None) == (sample_tree_abundances_train is None)
        assert (sample_tree_abundances_test is None) == (sample_tree_abundances_test is None)
        if sample_tree_generation_functions is None:
            self.sample_tree_generation_functions = []
            for k in range(len(train_sample_pack)):
                self.sample_tree_generation_functions.append({
                    "train_sample": train_sample_pack[k],
                    "test_sample": test_sample_pack[k],
                    "train_square_sample": train_square_sample_pack[k],
                    "test_square_sample": test_square_sample_pack[k]
                })

            self.sample_tree_generation_functions.append({
                "train_sample": data_bert.obtain_train_sample,
                "test_sample": data_bert.obtain_test_sample,
                "train_square_sample": data_bert.obtain_train_square_sample,
                "test_square_sample": data_bert.obtain_test_square_sample
            })
            self.sample_tree_verification_functions = has_tree_correlations_pack.copy()
            self.sample_tree_verification_functions.append(data_bert.has_correlations)
            # the functions to verify whether a tuple (content, topic) has correlations.

            self.sample_tree_abundances_train = data_bert_tree_struct.topic_trees_filtered_abundances_train
            self.sample_tree_abundances_test = data_bert_tree_struct.topic_trees_filtered_abundances_test
            self.generation_sizes = [4, 12, 14, 25, 25]
        else:
            assert len(sample_tree_generation_functions) == len(sample_tree_verification_functions)
            assert len(sample_tree_verification_functions)-1 == len(sample_tree_abundances_train)
            assert len(sample_tree_abundances_train) == len(sample_tree_abundances_test)
            assert len(sample_tree_abundances_test) == len(generation_sizes)

            self.sample_tree_generation_functions = sample_tree_generation_functions
            self.sample_tree_verification_functions = sample_tree_verification_functions
            self.sample_tree_abundances_train = sample_tree_abundances_train
            self.sample_tree_abundances_test = sample_tree_abundances_test
            self.generation_sizes = generation_sizes

        # normalization factor, if the (topics_subtree, contents) in each level are sampled once only.
        self.train_raw_normalization_factor = self.sample_tree_abundances_train.astype(dtype=np.float32) / len(data_bert.train_topics_num_id)
        self.test_raw_normalization_factor = self.sample_tree_abundances_test.astype(dtype=np.float32) / len(data_bert.test_topics_num_id)

        # we further normalize by the number of draws
        no_draws = np.array([(size*size) for size in self.generation_sizes])
        self.train_actual_normalization_factor = self.train_raw_normalization_factor / np.array(no_draws)
        self.test_actual_normalization_factor = self.test_raw_normalization_factor / np.array(no_draws)

        # boosting multiplier per level. this is to make sure the higher the level the more the domination.
        # the sum is around 5000, so a combo with sample_size = 10000 should dominate this in turn should have
        # zero-one ratio 2:1
        per_level_boosting_factor = np.array([1, 1.5, 2.25, 3.375, 5.0625], dtype=np.float32) * 379
        self.train_multiplier = self.train_actual_normalization_factor * per_level_boosting_factor
        self.test_multiplier = self.test_actual_normalization_factor * per_level_boosting_factor

    # returns a sequence o 6-tuples, topics_num_id, contents_num_id, correlations,
    # classes, tree_levels, abundance_multipliers

    # the number of tree tcs are always square and of fixed size. the sample_size argument below
    # controls the size of the fine grained samples.
    def obtain_train_sample(self, sample_size):
        topics_ll, contents_ll, cors_ll = self.sample_tree_generation_functions[-1]["train_sample"](sample_size // 2, sample_size // 2)
        tree_level_sizes = [(size*size) for size in self.generation_sizes] # how much samples per level
        tree_level_sizes.append(len(cors_ll))

        tree_level_multipliers = list(self.train_multiplier)
        tree_level_multipliers.append(1)

        tree_levels = np.repeat(np.arange(len(self.sample_tree_generation_functions)), np.array(tree_level_sizes))
        multipliers = np.repeat(tree_level_multipliers, np.array(tree_level_sizes))


        topics_tree = [] # per level list
        contents_tree = []
        cors_tree = []

        start = 0
        for k in range(len(self.sample_tree_abundances_train)):
            tps, cts, crs = self.sample_tree_generation_functions[k]["train_square_sample"](self.generation_sizes[k])
            topics_tree.append(tps)
            contents_tree.append(cts)
            cors_tree.append(crs)

            multipliers[start:(start+len(cts))] = multipliers[start:(start+len(cts))] * np.array(
                [len(data_bert_tree_struct.topics_group_filtered[k]["group_filter_available"][topic_tree_klevel_id]) for topic_tree_klevel_id in tps]
            )

            assert (tree_levels[start:(start+len(cts))] != k).sum() == 0
            start = start + len(cts)
        topics = np.concatenate([np.concatenate(topics_tree), topics_ll])
        contents = np.concatenate([np.concatenate(contents_tree), contents_ll])
        cors = np.concatenate([np.concatenate(cors_tree), cors_ll])

        return topics, contents, cors, None, tree_levels, multipliers

    # returns a sequence o 6-tuples, topics_num_id, contents_num_id, correlations,
    # classes, tree_levels, abundance_multipliers

    # the number of tree tcs are always square and of fixed size. the sample_size argument below
    # controls the size of the fine grained samples.
    def obtain_test_sample(self, sample_size):
        topics_ll, contents_ll, cors_ll = self.sample_tree_generation_functions[-1]["test_sample"](sample_size // 2,
                                                                                                    sample_size // 2)
        tree_level_sizes = [(size * size) for size in self.generation_sizes]  # how much samples per level
        tree_level_sizes.append(len(cors_ll))

        tree_level_multipliers = list(self.test_multiplier)
        tree_level_multipliers.append(1)

        tree_levels = np.repeat(np.arange(len(self.sample_tree_generation_functions)), np.array(tree_level_sizes))
        multipliers = np.repeat(tree_level_multipliers, np.array(tree_level_sizes))

        topics_tree = []  # per level list
        contents_tree = []
        cors_tree = []

        start = 0
        for k in range(len(self.sample_tree_abundances_test)):
            tps, cts, crs = self.sample_tree_generation_functions[k]["test_square_sample"](self.generation_sizes[k])
            topics_tree.append(tps)
            contents_tree.append(cts)
            cors_tree.append(crs)

            multipliers[start:(start + len(cts))] = multipliers[start:(start + len(cts))] * np.array(
                [len(data_bert_tree_struct.topics_group_filtered[k]["group_filter_available"][topic_tree_klevel_id]) for
                 topic_tree_klevel_id in tps]
            )

            assert (tree_levels[start:(start + len(cts))] != k).sum() == 0
            start = start + len(cts)
        topics = np.concatenate([np.concatenate(topics_tree), topics_ll])
        contents = np.concatenate([np.concatenate(contents_tree), contents_ll])
        cors = np.concatenate([np.concatenate(cors_tree), cors_ll])

        return topics, contents, cors, None, tree_levels, multipliers

    def obtain_train_notree_sample(self, sample_size):
        topics_ll, contents_ll, cors_ll = self.sample_tree_generation_functions[-1]["train_sample"](sample_size // 2,
                                                                                                    sample_size // 2)

        tree_levels = np.repeat(len(self.sample_tree_generation_functions)-1, len(cors_ll))
        multipliers = np.repeat(1, len(cors_ll))

        return topics_ll, contents_ll, cors_ll, None, tree_levels, multipliers

    def obtain_test_notree_sample(self, sample_size):
        topics_ll, contents_ll, cors_ll = self.sample_tree_generation_functions[-1]["test_sample"](sample_size // 2,
                                                                                                    sample_size // 2)

        tree_levels = np.repeat(len(self.sample_tree_generation_functions)-1, len(cors_ll))
        multipliers = np.repeat(1, len(cors_ll))

        return topics_ll, contents_ll, cors_ll, None, tree_levels, multipliers

    def obtain_train_square_sample(self, sample_size):
        raise Exception("This function is not supported for tree samplers.")

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        raise Exception("This function is not supported for tree samplers.")

    def obtain_tree_train_sample(self, sample_size, level):
        return self.sample_tree_generation_functions[level]["train_sample"](sample_size // 2, sample_size // 2)

    def obtain_tree_test_sample(self, sample_size, level):
        return self.sample_tree_generation_functions[level]["test_sample"](sample_size // 2, sample_size // 2)

    def obtain_tree_train_square_sample(self, sample_size, level):
        return self.sample_tree_generation_functions[level]["train_square_sample"](int(math.sqrt(sample_size)))

    def obtain_tree_test_square_sample(self, sample_size, level):
        return self.sample_tree_generation_functions[level]["test_square_sample"](int(math.sqrt(sample_size)))

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        raise Exception("This function is not supported for tree samplers.")

    # check if there are any correlations at the individual (last) level.
    def has_correlations_individual(self, content_num_ids, topic_num_ids):
        return self.sample_tree_verification_functions[-1](content_num_ids, topic_num_ids)

    def has_correlations_tree_level(self, content_num_ids, topic_levelk_ids, level):
        return self.sample_tree_verification_functions[level](content_num_ids, topic_levelk_ids)

    def is_tree_sampler(self):
        return True

class ReversedTreeSampler(SamplerBase):
    # tree sampler, but reverses train and test sets.
    def __init__(self, tree_sampler):
        SamplerBase.__init__(self)
        assert tree_sampler.is_tree_sampler()
        self.tree_sampler = tree_sampler

    def obtain_train_sample(self, sample_size):
        return self.tree_sampler.obtain_test_sample(sample_size)

    # returns a sequence o 6-tuples, topics_num_id, contents_num_id, correlations,
    # classes, tree_levels, abundance_multipliers

    # the number of tree tcs are always square and of fixed size. the sample_size argument below
    # controls the size of the fine grained samples.
    def obtain_test_sample(self, sample_size):
        return self.tree_sampler.obtain_train_sample(sample_size)

    def obtain_train_notree_sample(self, sample_size):
        return self.tree_sampler.obtain_test_notree_sample(sample_size)

    def obtain_test_notree_sample(self, sample_size):
        return self.tree_sampler.obtain_train_notree_sample(sample_size)

    def obtain_train_square_sample(self, sample_size):
        raise Exception("This function is not supported for tree samplers.")

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        raise Exception("This function is not supported for tree samplers.")

    def obtain_tree_train_sample(self, sample_size, level):
        return self.tree_sampler.obtain_tree_test_sample(sample_size, level)

    def obtain_tree_test_sample(self, sample_size, level):
        return self.tree_sampler.obtain_tree_train_sample(sample_size, level)

    def obtain_tree_train_square_sample(self, sample_size, level):
        return self.tree_sampler.obtain_tree_test_square_sample(sample_size, level)

    def obtain_tree_test_square_sample(self, sample_size, level):
        return self.tree_sampler.obtain_tree_train_square_sample(sample_size, level)

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        raise Exception("This function is not supported for tree samplers.")

    # check if there are any correlations at the individual (last) level.
    def has_correlations_individual(self, content_num_ids, topic_num_ids):
        return self.tree_sampler.has_correlations_individual(content_num_ids, topic_num_ids)

    def has_correlations_tree_level(self, content_num_ids, topic_levelk_ids, level):
        return self.tree_sampler.has_correlations_tree_level(content_num_ids, topic_levelk_ids, level)

    def is_tree_sampler(self):
        return True

# This class is to draw mixed samples from the
class MixedSampler(SamplerBase):
    # draws from a list of samplers, with given probabilities. If sampler_probas is none, assumes uniformly distributed
    def __init__(self, sampler_list, sampler_probas = None):
        SamplerBase.__init__(self)
        self.sampler_list = sampler_list
        if sampler_probas is None:
            self.sampler_probas = list(np.repeat(1 / len(sampler_list), len(sampler_list)))
        else:
            assert len(sampler_probas) == len(sampler_list)
            assert abs(sum(sampler_probas) - 1) < 0.00001
            self.sampler_probas = sampler_probas

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_sample(self, sample_size):
        sample_topics = []
        sample_contents = []
        sample_cors = []
        sample_class_ids = []
        for k in range(len(self.sampler_probas)):
            proba = self.sampler_probas[k]
            sampler = self.sampler_list[k]
            topics, contents, cors, class_ids = sampler.obtain_train_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(sample_cors), np.concatenate(sample_class_ids)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        sample_topics = []
        sample_contents = []
        sample_cors = []
        sample_class_ids = []
        for k in range(len(self.sampler_probas)):
            proba = self.sampler_probas[k]
            sampler = self.sampler_list[k]
            topics, contents, cors, class_ids = sampler.obtain_test_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(sample_cors), np.concatenate(
            sample_class_ids)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_train_square_sample(self, sample_size):
        sample_topics = []
        sample_contents = []
        sample_cors = []
        sample_class_ids = []
        for k in range(len(self.sampler_probas)):
            proba = self.sampler_probas[k]
            sampler = self.sampler_list[k]
            topics, contents, cors, class_ids = sampler.obtain_train_square_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(sample_cors), np.concatenate(
            sample_class_ids)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_square_sample(self, sample_size):
        sample_topics = []
        sample_contents = []
        sample_cors = []
        sample_class_ids = []
        for k in range(len(self.sampler_probas)):
            proba = self.sampler_probas[k]
            sampler = self.sampler_list[k]
            topics, contents, cors, class_ids = sampler.obtain_test_square_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(sample_cors), np.concatenate(
            sample_class_ids)

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        if class_ids is None:
            return self.sampler_list[0].sample_verification_function(content_num_ids, topic_num_ids)
        cors = np.zeros(shape = len(content_num_ids))
        for k in range(len(self.sampler_probas)):
            k_class_locations = (class_ids == k)
            cors[k_class_locations] = self.sampler_list[k].has_correlations(content_num_ids[k_class_locations], topic_num_ids[k_class_locations], None)
        return cors

default_sampler_instance = DefaultSampler()
default_sampler_overshoot2_instance = DefaultSampler(sample_generation_functions = {
                                                        "train_sample": data_bert_tree_struct.obtain_train_sample,
                                                        "test_sample": data_bert_tree_struct.obtain_test_sample,
                                                        "train_square_sample": data_bert_tree_struct.obtain_train_square_sample,
                                                        "test_square_sample": data_bert_tree_struct.obtain_test_square_sample
                                                    },
                                                     sample_verification_function = data_bert_tree_struct.has_close_correlations)
default_sampler_overshoot3_instance = DefaultSampler(sample_generation_functions = {
                                                        "train_sample": data_bert_tree_struct.obtain_further_train_sample,
                                                        "test_sample": data_bert_tree_struct.obtain_further_test_sample,
                                                        "train_square_sample": data_bert_tree_struct.obtain_further_train_square_sample,
                                                        "test_square_sample": data_bert_tree_struct.obtain_further_test_square_sample
                                                    },
                                                     sample_verification_function = data_bert_tree_struct.has_further_correlations)

if data_bert_hacky_disable_tree_struct.enable_tree_struct:
    default_tree_sampler_instance = DefaultTreeSampler()

    overshoot_generation = default_tree_sampler_instance.sample_tree_generation_functions.copy()
    overshoot_generation[-1] = {
                    "train_sample": data_bert_tree_struct.obtain_train_sample,
                    "test_sample": data_bert_tree_struct.obtain_test_sample,
                    "train_square_sample": data_bert_tree_struct.obtain_train_square_sample,
                    "test_square_sample": data_bert_tree_struct.obtain_test_square_sample
                }
    overshoot_veri = default_tree_sampler_instance.sample_tree_verification_functions.copy()
    overshoot_veri[-1] = data_bert_tree_struct.has_close_correlations
    default_tree_overshoot_sampler_instance = DefaultTreeSampler(sample_tree_generation_functions=overshoot_generation, sample_tree_verification_functions=overshoot_veri, sample_tree_abundances_train=data_bert_tree_struct.topic_trees_filtered_abundances_train,
                                                                 sample_tree_abundances_test=data_bert_tree_struct.topic_trees_filtered_abundances_test, generation_sizes=[4, 9, 10, 17, 17])

default_reversed_sampler_instance = ReversedSampler(default_sampler_instance)
default_reversed_sampler_overshoot2_instance = ReversedSampler(default_sampler_overshoot2_instance)
default_reversed_sampler_overshoot3_instance = ReversedSampler(default_sampler_overshoot3_instance)

if data_bert_hacky_disable_tree_struct.enable_tree_struct:
    default_reversed_tree_sampler = ReversedTreeSampler(default_tree_sampler_instance)
    default_reversed_tree_sampler_overshoot = ReversedTreeSampler(default_tree_overshoot_sampler_instance)