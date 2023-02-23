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
