import model_bert_fix
import data_bert
import config
import time
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

print("premodel")
model = model_bert_fix.Model(units_size = 450)
print("presampler")
training_sampler = model_bert_fix.TrainingSampler(embedded_vectors_folder = config.resources_path + "bert_embedded_vectors/bert_vectorized_L6_H128/",
                               contents_one_hot_file = config.resources_path + "one_hot_languages/contents_lang_train.npy",
                               topics_one_hot_file = config.resources_path + "one_hot_languages/topics_lang_train.npy", device = "cpu")
print("postsampler")
model.compile()
model.set_training_params(1000, 1000, training_sampler = training_sampler, training_max_size = 100000)

ctime = time.time()
checkpoint_file = config.training_models_path + "model_checkpoints/{epoch:07d}.ckpt"
logging_file = config.training_models_path + "model_checkpoints/logfile.csv"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file, save_weights_only = True, verbose = 0, save_freq = 20)
csv_logger = tf.keras.callbacks.CSVLogger(logging_file, separator=',', append=False)

hist = model.fit(np.array([1, 2]), epochs = 10, callbacks=[callback, csv_logger], verbose = 2, steps_per_epoch = 1)
ctime = time.time() - ctime
print(ctime)

"""running_data = pd.DataFrame(hist.history)
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