# How to open notebook in your folder.

You probably know how to use conda. First open anaconda prompt

    conda activate /path/your/environment

And then CD to your working directory

    cd PycharmProjects/KaggleLearningEquality2023/jason

Run the jupyter notebook now

    python -m notebook


# 2023-02-23 TODO:

There are 3 models in the models folder. Help me find these few information:
* For each model, do the following:
    * Epoch of optimal test_precision vs test_recall (where both are equally large)
    * Epoch of optimal test_precision vs test_recall, but test_recall is more important.
        * This means we want to find the largest test_recall, such that test_precision $>0.7$
        * Also find the largest test_recall for test_precision $>0.75$
    * For the above 3 epochs, create a table such that all other information (columns) in those epochs are included
    * Plot a graph with test_accuracy, test_recall, test_precision, test_entropy, test_square_precision, test_square_recall for all epochs
        * Include the previous 3 epoch you found as vertical lines in the graph.
* After you are done, place it in a Word document **including the date of the models** so we can keep a record.

# 2023-02-24 TODO:

There are 7 more models in the models folder. Git fetch -> git pull to download them.
* Apply the *previous* tasks as for the *previous* 3 models to each of the new 7 models.
* For each 7 new models, include a plot of test_accuracy, test_recall, test_precision, test_entropy, test_overshoot_recall, test_overshoot_precision
* Among all 10 models, find the one which has the "best" balanced test_precision vs test_recall ratio
    * (which means the epoch where $\text{test\\_precision} \approx \text{test\\_recall}$, and we find the model which has the highest)
    
# 2023-03-01 TODO:

There will be a bunch of models to analyze. Add return values to your previously defined function to obtain the optimal epoch, and optimal values for test_recall and test_precision for a model. Call your function for each new model, and find the model which performs the best. Find the corresponding weight_decay and noise.

# 2023-03-04 TODO:
There are some predicted data for you to compare with the real data. The real data is in 

    ../data/contents_translate.csv
    ../data/topics_translate.csv
    ../data/correlations.csv

You should extract the predicted data into the jason folder (topics_tree_train.zip, topics_tree_test.zip). These are the predicted correlations by the model. The *actual* correlations are stored in correlations.csv, and the *predicted* correlations are stored in the zip files. 

### correlations.csv format
This file is given by the competition, containing the information of the *actual* contents that each topic contains. This file is stored in csv format, so we should use pandas (pd) to read from it. The topic id and content id are stored in **string** id format, e.g. t_xxxxxxxxx for topic, c_xxxxxxxxx for content. 

### predicted from model format
This zip file contains a bunch of .npy files. Each .npy file starts with an integer, and the integer corresponds to the topic in **int** id format. For each .npy file, you can use

    np.load(<file>)
    
to load the integer array into Python. The array contains all the predicted contents from the model (in **int** id) belonging to the topic. You *should* find a way to convert from **string** id to **int** id to compare the *true* correlations vs *predicted* correlations. Hint: load contents_translate.csv using pandas and use .index.

If you forget what the tree structure means, you can run tree_structure_visualization.py

### Requirements:
Help compute the following metrics:

$$
\begin{gather*}
\text{True positive} = \text{Predicted to be true and correct (actually true)}   \\
\text{False positive} = \text{Predicted to be true and wrong (actually false)}   \\
\text{True negative} = \text{Predicted to be false and correct (actually false)}   \\
\text{False negative} = \text{Predicted to be false and wrong (actually true)}   \\
\text{Precision} = \frac{\text{True positive}}{\text{True positive} + \text{False positive}}   \\
\text{Recall} = \frac{\text{True positive}}{\text{True positive} + \text{False negative}}   \\
F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}
\end{gather*}
$$

Apart from computing these for the whole set, we compute the rowwise (topic-wise) versions also. This means

$$
\begin{align*}
\text{Precision}_{r} &= \frac{1}{|\text{rows}|}\sum_{\text{row} \in \text{rows}} \frac{\text{True positive}_\text{row}}{\text{True positive}_\text{row} + \text{False positive}_\text{row}} \\
&= \frac{1}{|\text{topics}|}\sum_{\text{topic} \in \text{topics}} \frac{\text{True positive}_\text{topic}}{\text{True positive}_\text{topic} + \text{False positive}_\text{topic}}
\end{align*}
$$

and so on....

Also, compute the correlations **with respect to tree structure**, such that there exist a *tree correlation* between $(T,C)$ if and only if there exists a subnode $T'$ of $T$ such that $(T',C)$ is a *usual correlation*. Compute the above scores with respect to these correlations.
