import model_bert_fix
import data_bert
import data_bert_tree_struct
import config
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import os
# from IPython.display import Markdown
# import matplotlib.pyplot as plt

def train_model(model_name, custom_metrics = None, custom_stopping_func = None, custom_generation_functions = None):
    model = model_bert_fix.Model(units_size = 512)
    modeldir = config.training_models_path + model_name

    training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "bert_embedded_vectors/bert_vectorized_L6_H128/",
                                   contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                                   topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
    print("postsampler")
    model.compile()
    if custom_metrics is None:
        custom_metrics = model_bert_fix.DefaultMetrics()
    if custom_stopping_func is None:
        custom_stopping_func = model_bert_fix.DefaultStoppingFunc(modeldir)
    model.set_training_params(7500, 7500, training_sampler = training_sampler, training_max_size = 75000, custom_metrics = custom_metrics, custom_stopping_func = custom_stopping_func, custom_generation_functions = custom_generation_functions)

    if not os.path.isdir(modeldir + "/"):
        os.mkdir(modeldir + "/")

    ctime = time.time()
    checkpoint_file = modeldir + "/{epoch:07d}.ckpt"
    logging_file = modeldir + "/logfile.csv"
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = False, verbose = 0, save_freq = 20)
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

    hist = model.fit(np.array([1, 2]), epochs = 3000, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
    ctime = time.time() - ctime
    print(ctime)

    model.save_weights(modeldir + "/final_epoch.ckpt")

    del model, training_sampler

# train_model("direct_model")

def obtain_half_train_sample(one_sample_size, zero_sample_size):
    topics, contents, cors = data_bert.obtain_train_sample(one_sample_size // 2, zero_sample_size // 2)
    topics2, contents2, cors2 = data_bert_tree_struct.obtain_train_sample(one_sample_size // 2, zero_sample_size // 2)
    return np.concatenate([topics, topics2], axis = 0), np.concatenate([contents, contents2], axis = 0), np.concatenate([cors, cors2], axis = 0)

def obtain_further_half_train_sample(one_sample_size, zero_sample_size):
    topics, contents, cors = data_bert_tree_struct.obtain_further_train_sample(one_sample_size // 3, zero_sample_size // 3)
    topics2, contents2, cors2 = data_bert_tree_struct.obtain_train_sample(one_sample_size // 3, zero_sample_size // 3)
    topics3, contents3, cors3 = data_bert.obtain_train_sample(one_sample_size // 3, zero_sample_size // 3)
    return np.concatenate([topics, topics2, topics3], axis = 0), np.concatenate([contents, contents2, contents3], axis = 0), np.concatenate([cors, cors2, cors3], axis = 0)

"""metrics = model_bert_fix.OvershootMetrics()
sample_generation_functions = {
    "train_sample": obtain_half_train_sample,
    "test_sample": data_bert_tree_struct.obtain_test_sample,
    "train_square_sample": data_bert_tree_struct.obtain_train_square_sample,
    "test_square_sample": data_bert_tree_struct.obtain_test_square_sample
}
train_model("overshoot3", custom_metrics = metrics, custom_generation_functions = sample_generation_functions)
"""
metrics = model_bert_fix.OvershootMetrics()
sample_generation_functions = {
    "train_sample": obtain_further_half_train_sample,
    "test_sample": data_bert_tree_struct.obtain_further_test_sample,
    "train_square_sample": data_bert_tree_struct.obtain_further_train_square_sample,
    "test_square_sample": data_bert_tree_struct.obtain_further_test_square_sample
}
train_model("overshoot5", custom_metrics = metrics, custom_generation_functions = sample_generation_functions)


"""del model
del training_sampler

running_data = pd.DataFrame(hist.history)
display(Markdown("**Entropy metrics**"))
running_data[["entropy", "entropy_large_set","full_entropy"]].plot(figsize = (30, 30))

display(Markdown("**Training metrics**"))
running_data[["accuracy", "precision", "recall"]].plot(figsize = (30, 30))
plt.show()

display(Markdown("**Training full metrics**"))
running_data[[x for x in running_data.columns if x.startswith("full")]].plot(figsize = (30, 30))
plt.show()

display(Markdown("**Test metrics**"))
running_data[[x for x in running_data.columns if x.startswith("test")]].plot(figsize = (30, 30))
plt.show()

display(Markdown("**Training full metrics (no lang)**"))
running_data[[x for x in running_data.columns if x.startswith("no_lang_full")]].plot(figsize = (30, 30))
plt.show()

display(Markdown("**Test metrics (no lang)**"))
running_data[[x for x in running_data.columns if x.startswith("no_lang_test")]].plot(figsize = (30, 30))
plt.show()"""