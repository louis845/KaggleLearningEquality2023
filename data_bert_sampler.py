# class to manage sampling and correlation verification. this class is the most general sampling from data method, which
# supports sampling from overshoot samples, tree samples and so on. also, it supports dividing the sampled parts into
# disjoint subsets, where the samples from each subset are drawn from different respective overshoot samples.
import math
import numpy as np
import data_bert
import data_bert_tree_struct


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
            topics, contents, cors = sampler.obtain_train_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(cors), np.concatenate(sample_class_ids)

    # returns a sequence o 4-tuples, topics_num_id, contents_num_id, correlations, classes.
    def obtain_test_sample(self, sample_size):
        sample_topics = []
        sample_contents = []
        sample_cors = []
        sample_class_ids = []
        for k in range(len(self.sampler_probas)):
            proba = self.sampler_probas[k]
            sampler = self.sampler_list[k]
            topics, contents, cors = sampler.obtain_test_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(cors), np.concatenate(
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
            topics, contents, cors = sampler.obtain_train_square_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(cors), np.concatenate(
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
            topics, contents, cors = sampler.obtain_test_square_sample(int(sample_size * proba))
            class_ids = np.repeat(k, len(cors))
            sample_topics.append(topics)
            sample_contents.append(contents)
            sample_cors.append(cors)
            sample_class_ids.append(class_ids)
        return np.concatenate(sample_topics), np.concatenate(sample_contents), np.concatenate(cors), np.concatenate(
            sample_class_ids)

    # given the (content_id, topic_id, class) tuple, determine whether or not there is a correlation between them.
    def has_correlations(self, content_num_ids, topic_num_ids, class_ids):
        cors = np.zeros(shape = len(class_ids))
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