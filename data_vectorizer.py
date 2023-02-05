from nltk.corpus import wordnet
import nltk.stem
import pandas as pd
import numpy as np
import data

lem = nltk.stem.WordNetLemmatizer()

def to_wordnet(x):
    if x.startswith("V"):
        return wordnet.VERB
    if x.startswith("N"):
        return wordnet.NOUN
    if x.startswith("J"):
        return wordnet.ADJ
    if x.startswith("R"):
        return wordnet.ADV
    return None

def lemmatize(x):
    ppos = to_wordnet(nltk.pos_tag([x])[0][1])
    if ppos is None:
        return ""
    return lem.lemmatize(x, ppos)

def lemmatize_sentence(x):
    wlist = []
    words = nltk.pos_tag(nltk.word_tokenize(x))
    for word in words:
        ppos = to_wordnet(word[1])
        if ppos is not None:
            wlist.append(lem.lemmatize(word[0], ppos).lower())
    return wlist

def transform_replace_line_breaks(x):
    if type(x) == float:
        return x
    return x.replace("\n", " ")

def transform_replace_symbols(x):
    if type(x) == float:
        return x
    return x.replace("-", " ").replace("_", " ")

# this file is generated by data_exploration_text.ipynb
word_freqs = pd.read_csv("data/word_freqs.csv", index_col = 0)
# filter high occurence words
word_freqs_filtered = word_freqs.loc[word_freqs["frequency"] >= 100]
inverse_word_mapping = pd.DataFrame(index = word_freqs_filtered["word"], data = word_freqs_filtered.index, columns = ["pos"])

# vectorization of a sentence
def vectorize(x):
    if type(x) == float:
        return []
    lemma_list = lemmatize_sentence(x.replace("\n", " ").replace("-", " ").replace("_", " "))
    veclist = inverse_word_mapping["pos"].loc[inverse_word_mapping.index.intersection(lemma_list)]
    return list(veclist)

learnable_contents = pd.read_csv("data/learnable_contents.csv", index_col = 0)
learnable_topics = pd.read_csv("data/learnable_topics.csv", index_col = 0)

def obtain_learnable_contents(channel_list):
    channels_contents = data.obtain_contents(channel_list)
    learnables_subseries = learnable_contents.loc[channels_contents]
    return learnables_subseries.loc[learnables_subseries["0"]].index

def obtain_learnable_topics(channel_list):
    topics = list(data.obtain_topics(channel_list).index)
    learnables_subseries = learnable_topics.loc[topics]
    return learnables_subseries.loc[learnables_subseries["0"]].index

def obtain_correlation_frame(topics_list, contents_list):
    arr = np.zeros(shape = (len(topics_list), len(contents_list)))
    cor_frame = pd.DataFrame(data = arr, index = list(topics_list), columns = list(contents_list))
    for topic_id, content_ids in data.correlations.loc[\
        list(set(topics_list).intersection(set(data.correlations.index)))\
    ].iterrows(): # restrict to the list of our contents
        for content_id in content_ids["content_ids"].split():
            if content_id in contents_list:
                cor_frame.loc[topic_id, content_id] = 1
    return cor_frame

def internal_transform(x):
    return pd.Series(name = x.name, data = [vectorize(x["title_translate"]), vectorize(x["description_translate"])], index = ["title_translate", "description_translate"])

def obtain_topics_vector(topics_list):
    return data.topics.loc[topics_list][["title_translate", "description_translate"]].apply(internal_transform, axis = 1)

def obtain_contents_vector(contents_list):
    return data.contents.loc[contents_list][["title_translate", "description_translate"]].apply(internal_transform, axis = 1)

# train data uses the largest connected component
train_contents = obtain_learnable_contents(data.channel_components[0])
train_topics = obtain_learnable_topics(data.channel_components[0])

def index_is_usable(idx):
    contents_count = len(obtain_learnable_contents(data.channel_components[idx]))
    topics = data.obtain_topics(data.channel_components[idx])
    learnable_subseries = learnable_topics.loc[list(topics.index)]
    topics_count = learnable_subseries["0"].sum()
    return contents_count > 100 and topics_count > 100

test_data_channels = []
for idx in range(1, len(data.channel_components)):
    if index_is_usable(idx):
        test_data_channels.extend(data.channel_components[idx])
test_contents = obtain_learnable_contents(test_data_channels)
test_topics = obtain_learnable_topics(test_data_channels)

def random_train_contents_sample(sample_size):
    return train_contents[list(np.random.choice(len(train_contents), sample_size, replace=False))]

def random_train_topics_sample(sample_size):
    return train_topics[list(np.random.choice(len(train_topics), sample_size, replace=False))]

def random_test_contents_sample(sample_size):
    return test_contents[list(np.random.choice(len(test_contents), sample_size, replace=False))]

def random_test_topics_sample(sample_size):
    return test_topics[list(np.random.choice(len(test_topics), sample_size, replace=False))]