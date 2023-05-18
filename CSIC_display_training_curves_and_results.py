import os

import matplotlib.pyplot as plt
import pandas as pd

# load saved_models/summary.csv
summary = pd.read_csv("saved_models/summary.csv", index_col = 0)
display(summary)

# iterate through each png file in saved_models folder, and display it individually with matplotlib.
for file in os.listdir("saved_models"):
    if file.endswith(".png"):
        plt.figure()
        plt.imshow(plt.imread("saved_models/" + file))
        plt.show()

# iterate through each subfolder in saved_models folder, and display the training_curve.png file with matplotlib.
for folder in os.listdir("saved_models"):
    if os.path.isdir("saved_models/" + folder):
        plt.figure()
        plt.imshow(plt.imread("saved_models/" + folder + "/training_curve.png"))
        plt.show()