import data_bert
import data_bert_sampler
import data_bert_tree_struct
import model_bert_fix
import model_bert_fix_stepup
import config
import time
import tensorflow as tf
import tensorflow_models as tfm
import numpy as np
import pandas as pd
import os
import gc
import sys
# from IPython.display import Markdown
# import matplotlib.pyplot as plt

def train_model(model_name, custom_metrics = None, custom_stopping_func = None, custom_tuple_choice_sampler = None, init_noise = 0.05, weight_decay = 0.01):
    model = model_bert_fix.Model(units_size = 512, init_noise=init_noise)
    modeldir = config.training_models_path + model_name

    training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "sbert_vectors/mininet384/",
                                   contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                                   topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
    print("postsampler")
    model.compile(weight_decay = weight_decay)
    if custom_metrics is None:
        custom_metrics = model_bert_fix.default_metrics
    if custom_stopping_func is None:
        custom_stopping_func = model_bert_fix.DefaultStoppingFunc(modeldir)
    model.set_training_params(15000, training_sampler = training_sampler, training_max_size = 75000, custom_metrics = custom_metrics, custom_stopping_func = custom_stopping_func, custom_tuple_choice_sampler = custom_tuple_choice_sampler)

    if not os.path.isdir(modeldir + "/"):
        os.mkdir(modeldir + "/")

    ctime = time.time()
    checkpoint_file = modeldir + "/{epoch:07d}.ckpt"
    logging_file = modeldir + "/logfile.csv"
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = False, verbose = 0, save_freq = 5)
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

    hist = model.fit(np.array([1, 2]), epochs = 4000, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
    ctime = time.time() - ctime
    print(ctime)

    model.save_weights(modeldir + "/final_epoch.ckpt")

    del model, training_sampler

def train_model_stepup(model_name, custom_metrics = None, custom_stopping_func = None, custom_tuple_choice_sampler = None, custom_tuple_choice_sampler_overshoot = None,
                       init_noise_topics = 0.05, init_noise_overshoot_topics = 0.2, init_noise_contents = 0.05, init_noise_overshoot_contents = 0.2,
                       init_noise_lang = 0.2, init_noise_overshoot_lang = 0.3, weight_decay = 0.0005):
    model = model_bert_fix_stepup.Model(units_size = 512, init_noise_topics = init_noise_topics, init_noise_overshoot_topics = init_noise_overshoot_topics,
                                        init_noise_contents = init_noise_contents, init_noise_overshoot_contents = init_noise_overshoot_contents,
                                        init_noise_lang = init_noise_lang, init_noise_overshoot_lang = init_noise_overshoot_lang)
    modeldir = config.training_models_path + model_name
    checkpoint_file = modeldir + "/{epoch:07d}.ckpt"
    logging_file = modeldir + "/logfile.csv"

    training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "sbert_vectors/mininet_L12_english384/",
                                   contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                                   topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
    print("postsampler")
    model.compile(weight_decay = weight_decay, learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.0005, decay_steps = 5000, alpha = 0.1)) # 0.0005
    if custom_metrics is None:
        custom_metrics = model_bert_fix_stepup.default_metrics
    if custom_stopping_func is None:
        custom_stopping_func = model_bert_fix_stepup.DefaultStoppingFunc(modeldir)
    model.set_training_params(12500, training_sampler = training_sampler, training_max_size = 12500, custom_metrics = custom_metrics, custom_stopping_func = custom_stopping_func,
                              custom_tuple_choice_sampler = custom_tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = custom_tuple_choice_sampler_overshoot)

    if not os.path.isdir(modeldir + "/"):
        os.mkdir(modeldir + "/")

    ctime = time.time()

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = False, verbose = 0, save_freq = 10)
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

    hist = model.fit(np.array([1, 2]), epochs = 6000, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
    ctime = time.time() - ctime
    print("finished first stepup time")
    print(ctime)

    """custom_stopping_func.force_final_state()
    model.state_is_final = True

    ctime = time.time()
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=True)
    model.compile(weight_decay = 0.001, learning_rate = tfm.optimization.lr_schedule.LinearWarmup(
        warmup_learning_rate = 0,
        after_warmup_lr_sched = tf.keras.optimizers.schedules.CosineDecay(0.002, decay_steps = 7000, alpha = 0.25),
        warmup_steps = 2500
    ))
    model.set_training_params(15000, training_sampler=training_sampler, training_max_size=75000,
                              custom_metrics=custom_metrics, custom_stopping_func=custom_stopping_func,
                              custom_tuple_choice_sampler=custom_tuple_choice_sampler,
                              custom_tuple_choice_sampler_overshoot=custom_tuple_choice_sampler_overshoot)
    model.fit(np.array([1, 2]), initial_epoch=100, epochs=4000, callbacks=[callback, csv_logger], verbose=2, steps_per_epoch=1) # replace 100 with len(hist.history["accuracy"])
    ctime = time.time() - ctime
    print("finished second stepup time")
    print(ctime)"""

    model.save_weights(modeldir + "/final_epoch.ckpt")

    del model, training_sampler

def train_model_stepup_small(model_name, custom_metrics = None, custom_stopping_func = None, custom_tuple_choice_sampler = None, custom_tuple_choice_sampler_overshoot = None,
                       init_noise_topics = 0.05, init_noise_overshoot_topics = 0.2, init_noise_contents = 0.05, init_noise_overshoot_contents = 0.2,
                       init_noise_lang = 0.2, init_noise_overshoot_lang = 0.3, weight_decay = 0.0005):
    model = model_bert_fix_stepup.Model(units_size = 256, init_noise_topics = init_noise_topics, init_noise_overshoot_topics = init_noise_overshoot_topics,
                                        init_noise_contents = init_noise_contents, init_noise_overshoot_contents = init_noise_overshoot_contents,
                                        init_noise_lang = init_noise_lang, init_noise_overshoot_lang = init_noise_overshoot_lang)
    modeldir = config.training_models_path + model_name
    checkpoint_file = modeldir + "/{epoch:07d}.ckpt"
    logging_file = modeldir + "/logfile.csv"

    training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "sbert_vectors/mininet_L12_english384/",
                                   contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                                   topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
    print("postsampler")
    model.compile(weight_decay = weight_decay, learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.0005, decay_steps = 5000, alpha = 0.1)) # 0.0005
    if custom_metrics is None:
        custom_metrics = model_bert_fix_stepup.default_metrics
    if custom_stopping_func is None:
        custom_stopping_func = model_bert_fix_stepup.DefaultStoppingFunc(modeldir)
    model.set_training_params(12500, training_sampler = training_sampler, training_max_size = 12500, custom_metrics = custom_metrics, custom_stopping_func = custom_stopping_func,
                              custom_tuple_choice_sampler = custom_tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = custom_tuple_choice_sampler_overshoot)

    if not os.path.isdir(modeldir + "/"):
        os.mkdir(modeldir + "/")

    ctime = time.time()

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = False, verbose = 0, save_freq = 10)
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

    hist = model.fit(np.array([1, 2]), epochs = 6000, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
    ctime = time.time() - ctime
    print(ctime)
    model.save_weights(modeldir + "/final_epoch.ckpt")

    del model, training_sampler

def train_model_stepup_dimreduce(model_name, custom_metrics = None, custom_stopping_func = None, custom_tuple_choice_sampler = None, custom_tuple_choice_sampler_overshoot = None,
                       init_noise_topics = 0.05, init_noise_overshoot_topics = 0.2, init_noise_contents = 0.05, init_noise_overshoot_contents = 0.2,
                       init_noise_lang = 0.2, init_noise_overshoot_lang = 0.3, weight_decay = 0.0005):
    model = model_bert_fix_stepup_dimreduce.Model(units_size = 512, init_noise_topics = init_noise_topics, init_noise_overshoot_topics = init_noise_overshoot_topics,
                                        init_noise_contents = init_noise_contents, init_noise_overshoot_contents = init_noise_overshoot_contents,
                                        init_noise_lang = init_noise_lang, init_noise_overshoot_lang = init_noise_overshoot_lang)
    modeldir = config.training_models_path + model_name
    checkpoint_file = modeldir + "/{epoch:07d}.ckpt"
    logging_file = modeldir + "/logfile.csv"

    training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "sbert_vectors/mininet_L12_english384/",
                                   contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                                   topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
    print("postsampler")
    model.compile(weight_decay = weight_decay, learning_rate = tf.keras.optimizers.schedules.CosineDecay(0.0005, decay_steps = 5000, alpha = 0.1)) # 0.0005
    if custom_metrics is None:
        custom_metrics = model_bert_fix_stepup.default_metrics
    if custom_stopping_func is None:
        custom_stopping_func = model_bert_fix_stepup.DefaultStoppingFunc(modeldir)
    model.set_training_params(12500, training_sampler = training_sampler, training_max_size = 12500, custom_metrics = custom_metrics, custom_stopping_func = custom_stopping_func,
                              custom_tuple_choice_sampler = custom_tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = custom_tuple_choice_sampler_overshoot)

    if not os.path.isdir(modeldir + "/"):
        os.mkdir(modeldir + "/")

    ctime = time.time()

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = False, verbose = 0, save_freq = 10)
    csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

    hist = model.fit(np.array([1, 2]), epochs = 6000, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
    ctime = time.time() - ctime
    print("finished first stepup time")
    print(ctime)

    model.save_weights(modeldir + "/final_epoch.ckpt")

    del model, training_sampler

"""tuple_choice_sampler = data_bert_sampler.MixedSampler(sampler_list = [data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance])
metrics = model_bert_fix.obtain_overshoot_metric_instance(tuple_choice_sampler)
train_model("overshoot2", custom_metrics = metrics, custom_tuple_choice_sampler =  tuple_choice_sampler)

tuple_choice_sampler = data_bert_sampler.MixedSampler(sampler_list = [data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, data_bert_sampler.default_sampler_overshoot3_instance], sampler_probas=[0.5, 1.0/3, 1.0/6])
metrics = model_bert_fix.obtain_overshoot_metric_instance(tuple_choice_sampler)
train_model("overshoot23", custom_metrics = metrics, custom_tuple_choice_sampler =  tuple_choice_sampler)"""

train_model_stepup("minilm12_eng_model_stepup_precise_lnoise")

"""tuple_choice_sampler = data_bert_sampler.default_sampler_instance
tuple_choice_sampler_overshoot = data_bert_sampler.MixedSampler(sampler_list = [data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance])
metrics = model_bert_fix_stepup.obtain_overshoot_metric_instance(tuple_choice_sampler, tuple_choice_sampler_overshoot)
train_model_stepup("overshoot2_stepup_more_noise", custom_metrics = metrics, custom_tuple_choice_sampler = tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = tuple_choice_sampler_overshoot)

tuple_choice_sampler = data_bert_sampler.default_sampler_instance
tuple_choice_sampler_overshoot = data_bert_sampler.MixedSampler(sampler_list = [data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot3_instance])
metrics = model_bert_fix_stepup.obtain_overshoot_metric_instance(tuple_choice_sampler, tuple_choice_sampler_overshoot)
train_model_stepup("overshoot3_stepup_more_noise", custom_metrics = metrics, custom_tuple_choice_sampler = tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = tuple_choice_sampler_overshoot)

tuple_choice_sampler = data_bert_sampler.default_sampler_instance
tuple_choice_sampler_overshoot = data_bert_sampler.MixedSampler(sampler_list = [data_bert_sampler.default_sampler_instance, data_bert_sampler.default_sampler_overshoot2_instance, data_bert_sampler.default_sampler_overshoot3_instance])
metrics = model_bert_fix_stepup.obtain_overshoot_metric_instance(tuple_choice_sampler, tuple_choice_sampler_overshoot)
train_model_stepup("overshoot23_stepup_more_noise", custom_metrics = metrics, custom_tuple_choice_sampler = tuple_choice_sampler, custom_tuple_choice_sampler_overshoot = tuple_choice_sampler_overshoot)"""

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