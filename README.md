# KaggleLearningEquality2023
Repo for the Learning Equality Kaggle competitions

# Description
## Tree structure visualization
Relevant files:
 * data_exploration_tree.ipynb
   * Investigates properties of tree structure
   * Properties of contents depending on tree structure
 * tree_structure_visualization.py
   * GUI to navigate the trees
   * Display the contents belonging to a node
 ## Python helper scripts
Relevant files:
 * data.py
   * Loads the translated Kaggle Learning Equality dataset
   * Creates a list of trees (topic_trees)
   * Contains pandas dataframe (contents, topics, correlations)
 # TODO
 ## Find 'method' to define metric on string data
Ideas:
 * Vectorize each string into zeros and ones
   * Zero represents word not exists, one represents word exists
   * Use NLTK WordNet for grouping (lemmatization)
 * Use pretrained transformer models (BERT)
   * https://www.tensorflow.org/text/tutorials/classify_text_with_bert
 ## Fit model for correlation tasks
Ideas:
 * K means, K nearest neighbors 
 * Decision tree on each topic tree
 