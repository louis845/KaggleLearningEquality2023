import data
import data_vectorizer
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

import model_vectorizer_simple

import numpy as np
import pandas as pd

model = model_vectorizer_simple.Model(zero_to_one_ratio = 0) # the ratio doesn't matter since it is used for training only.
model.compile()

# load the pretrained weights
pretrained_weights = ["./vectorizer_trained_weights/0001875.ckpt", "./vectorizer_trained_weights/0002592.ckpt", "./vectorizer_trained_weights/0003493.ckpt"]
model.load_weights("./vectorizer_trained_weights/0002592.ckpt")

# DEMO -- do some predictions with the model. The topics in the TEST set are data_vectorizer.test_topics and data_vectorizer.test_contents
sample_topics = data_vectorizer.test_topics[np.random.choice(len(data_vectorizer.test_topics), 5, replace=False)] # randomly choose 5 topics
sample_contents = data_vectorizer.test_contents[np.random.choice(len(data_vectorizer.test_contents), 5, replace=False)] # randomly choose 5 contents

# choose randomly topics that contain contents
sample_topics_with_contents = data.topics.loc[data_vectorizer.test_topics] # obtain the topics table restricted to the topics in TEST set
sample_topics_with_contents = sample_topics_with_contents.loc[sample_topics_with_contents["has_content"]].index # further restrict to those that have content
sample_topics_with_contents = sample_topics_with_contents[np.random.choice(len(sample_topics_with_contents), 25, replace = False)] # randomly choose 25 of them

print("Sample topics:   ", sample_topics)
print("Sample contents: ", sample_contents)

# obtain a sequence of tuples (sample topic id, sample content id)
sample_topic_sequence = sample_topics[np.tile(np.arange(5), 5)]
sample_contents_sequence = sample_contents[np.repeat(np.arange(5), 5)]

print("\nCartesian product:")
print(sample_topic_sequence)
print(sample_contents_sequence)

# add the sample_topics_with_contents to the sequences
for topic_id in sample_topics_with_contents:
    content_ids = data.correlations.loc[topic_id, "content_ids"] # data.correlations (data/correlations.csv) is the file for whether a topic contains some content
    content_ids = content_ids.split() # split with space " "
    # REMEMBER to restrict to test set
    content_ids = pd.Index(content_ids).intersection(data_vectorizer.test_contents)
    # choose a random content_id and add it to the sequence
    sample_topic_sequence =sample_topic_sequence.append(pd.Index([topic_id]))
    idx = np.random.choice(len(content_ids), 1)
    sample_contents_sequence = sample_contents_sequence.append(content_ids[idx])

print("\nTotal sequence to test:")
print(sample_topic_sequence)
print(sample_contents_sequence)