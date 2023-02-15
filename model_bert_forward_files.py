import numpy as np
import model_bert_fix
import config
import math
import data_bert
import tensorflow as tf
import gc
import os

def helper_eval_model(model, extractor, topics, contents):
    length = len(topics)
    result = np.zeros(shape = (length, 128))

    batch_size = 7500
    for k in range(int(math.ceil((length + 0.0) / batch_size))):
        low = k * batch_size
        high = min((k + 1) * batch_size, length)
        tlow = low
        thigh = high
        while tlow < high:
            try:
                mtopics = topics[np.arange(tlow, thigh)]
                mcontents = contents[np.arange(tlow, thigh)]
                input_dat = extractor.obtain_input_data(mtopics, mcontents)
                eval_result = model.eval_omit_last(input_dat)

                result[tlow:thigh, :] = tf.squeeze(eval_result, axis = 1).numpy()

                tlow = thigh
                thigh = high
            except RuntimeError as err:
                if not "CUDA failed with error out of memory" in str(err):
                    raise err
                thigh = max((thigh + tlow) // 2, tlow + 1)
                print("Too large batch! Decreasing size....")
    return result

def helper_eval_model_filter_lang(model, extractor, topics, contents):
    length = len(topics)
    result = np.zeros(shape = (length, 128))

    batch_size = 7500
    for k in range(int(math.ceil((length + 0.0) / batch_size))):
        low = k * batch_size
        high = min((k + 1) * batch_size, length)
        tlow = low
        thigh = high
        while tlow < high:
            try:
                mtopics = topics[np.arange(tlow, thigh)]
                mcontents = contents[np.arange(tlow, thigh)]
                input_dat = extractor.obtain_input_data_filter_lang(mtopics, mcontents)
                eval_result = model.eval_omit_last(input_dat)

                result[tlow:thigh, :] = tf.squeeze(eval_result, axis = 1).numpy()

                tlow = thigh
                thigh = high
            except RuntimeError as err:
                if not "CUDA failed with error out of memory" in str(err):
                    raise err
                thigh = max((thigh + tlow) // 2, tlow + 1)
                print("Too large batch! Decreasing size....")
    return result

def forward_files(embedded_vectors_folder, contents_one_hot_file, topics_one_hot_file, model_checkpoint_file, out_folder, units_size = 512):
    if not os.path.isdir(config.training_models_path + out_folder):
        os.mkdir(config.training_models_path + out_folder)
    print("Loading extractor.........")
    extractor = model_bert_fix.TrainingSampler(
        embedded_vectors_folder=embedded_vectors_folder,
        contents_one_hot_file=contents_one_hot_file,
        topics_one_hot_file=topics_one_hot_file, device = "cpu")
    print("Loading model for forwarding.........")
    model = model_bert_fix.Model(units_size = units_size)
    model.load_weights(config.training_models_path + model_checkpoint_file)
    print("Loaded model for forwarding")
    topics, contents, y = data_bert.obtain_train_sample(len(data_bert.train_contents), len(data_bert.train_contents))
    result = helper_eval_model(model, extractor, topics, contents)
    np.save(config.training_models_path + out_folder + "/train_vects.npy", result)
    np.save(config.training_models_path + out_folder + "/train_y.npy", y)
    result = helper_eval_model_filter_lang(model, extractor, topics, contents)
    np.save(config.training_models_path + out_folder + "/train_vects_nolang.npy", result)
    np.save(config.training_models_path + out_folder + "/train_y_nolang.npy", y)
    print("Finished training files")
    gc.collect()

    for k in range(10):
        topics, contents, y = data_bert.obtain_train_sample(len(data_bert.train_contents) // 4,
                                                            len(data_bert.train_contents) // 4)
        result = helper_eval_model(model, extractor, topics, contents)
        np.save(config.training_models_path + out_folder + "/test_train_vects" + str(k) + ".npy", result)
        np.save(config.training_models_path + out_folder + "/test_train_y" + str(k) + ".npy", y)
        result = helper_eval_model(model, extractor, topics, contents)
        np.save(config.training_models_path + out_folder + "/test_train_vects_nolang" + str(k) + ".npy", result)
        np.save(config.training_models_path + out_folder + "/test_train_y_nolang" + str(k) + ".npy", y)
        gc.collect()
    print("Finished test train files")
    for k in range(10):
        topics, contents, y = data_bert.obtain_test_sample(len(data_bert.test_contents) // 4,
                                                            len(data_bert.test_contents) // 4)
        result = helper_eval_model(model, extractor, topics, contents)
        np.save(config.training_models_path + out_folder + "/test_test_vects" + str(k) + ".npy", result)
        np.save(config.training_models_path + out_folder + "/test_test_y" + str(k) + ".npy", y)
        result = helper_eval_model(model, extractor, topics, contents)
        np.save(config.training_models_path + out_folder + "/test_test_vects_nolang" + str(k) + ".npy", result)
        np.save(config.training_models_path + out_folder + "/test_test_y_nolang" + str(k) + ".npy", y)
        gc.collect()
    print("Finished test test files")

print("Running program....")
forward_files(embedded_vectors_folder = config.resources_path + "bert_embedded_vectors/bert_vectorized_L6_H128/",
    contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
    topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy",
    model_checkpoint_file = "more_layers/0000020.ckpt", out_folder = "xgboosttesting", units_size = 425
)
print("Program ended successfully")