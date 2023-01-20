"""
Python file to manage / load the data fram the csv files. Defines classes such as trees, nodes for loading the tree structure.
"""


import pandas as pd

contents = pd.read_csv("data/content.csv", index_col = 0)
correlations = pd.read_csv("data/correlations.csv", index_col = 0)
topics = pd.read_csv("data/topics.csv", index_col = 0)

# find the unique ids for trees
topic_trees_id_list = list(topics["channel"].value_counts().index)

class Node:
    def __init__(self):
        self.parent = None
        self.children = []

class Node:
    def __init__(self, title, description, channel, category, level, language, has_content, uid):
        self.parent = None
        self.children = []
        self.title = title
        self.description = description
        self.channel = channel
        self.category = category
        self.level = level
        self.language = language
        self.has_content = has_content
        self.uid = uid

    def __str__(self):
        return "Topic: " + self.title + "   " + self.language

    def __del__(self):
        for child in self.children:
            del child
        del self.children

    def deep_copy(self):
        mcopy = Node(self.title, self.description, self.channel, self.category, self.level, self.language, self.has_content, self.uid)
        for child in self.children:
            child_copy = child.deep_copy()
            mcopy.children.append(child_copy)
            child_copy.parent = mcopy
        return mcopy

    # copies the tree into another tree structure (another class and copying function)
    # the copier must be a function which takes in a Node as an input and outputs its "equivalent" copy
    # the returned copy must belong to a class with attributes "parent" and "children" where "children" is a Python list
    def deep_copy_equivalent(self, copier):
        mcopy = copier(self)
        for child in self.children:
            child_copy = child.deep_copy_equivalent(copier)
            mcopy.children.append(child_copy)
            child_copy.parent = mcopy
        return mcopy

    # searches a node in preorder traversal, returns the first node satisfying the condition, if exists.
    # Returns a tuple (preorder_value, node) if such a node exists, otherwise returns None.
    # the condition should be a function which return True/False based on this node.
    # at - This parameter restricts the search on leaves or non-leaves. Either None, "leaf", "nonleaf"
    def search_preorder(self, condition, preorder_value = [0], at = None):
        # reset preorder index
        if self.level == 0:
            preorder_value[0] = 0
        cvalue = preorder_value[0]
        preorder_value[0] += 1
        if condition(self):
            if at == None:
                return (cvalue, self)
            elif at == "leaf" and len(self.children) == 0:
                return (cvalue, self)
            elif at == "nonleaf" and len(self.children) > 0:
                return (cvalue, self)

        for child in self.children:
            res = child.search_preorder(condition, preorder_value, at = at)
            if res is not None:
                return res
        return None


# dictionary of trees starting from root level. the key is the channel of the tree,
# and the value is the object for the root Node of the tree
topic_trees = dict()
# dictionary of all the nodes, key as UID of the topic, and value equal to Node object
topic_total_nodes = dict()
# dictionary of all contents belonging to a channel of a tree, key as UID of the channel,
# value equal to a list of content UIDs.
topic_trees_contents = dict()

def initialize_topic_trees():
    # loop through levels
    for k in range(topics["level"].max() + 1):
        # get all topics at level k
        topics_lvl_k = topics.loc[topics["level"] == k]
        for idx in topics_lvl_k.index:
            title = topics_lvl_k.loc[idx, "title"]
            description = topics_lvl_k.loc[idx, "description"]
            channel = topics_lvl_k.loc[idx, "channel"]
            category = topics_lvl_k.loc[idx, "category"]
            level = topics_lvl_k.loc[idx, "level"]
            language = topics_lvl_k.loc[idx, "language"]
            parent = topics_lvl_k.loc[idx, "parent"]
            has_content = topics_lvl_k.loc[idx, "has_content"]

            node = Node(title, description, channel, category, level, language, has_content, idx)
            topic_total_nodes[idx] = node
            # at root level iff k = 0
            if k == 0:
                # if at root level, we add the root nodes to topic_trees
                topic_trees[channel] = node
            else:
                # otherwise, we add the current node to the parent
                parent_node = topic_total_nodes[parent]
                parent_node.children.append(node)
                node.parent = parent_node

def initialize_topic_tree_contents():
    # first consider the topics subtable with only rows where has_content is True.
    topic_and_contents = topics.loc[topics.has_content == True]\
        .merge(correlations, left_on="id", right_on="topic_id", how="inner")
    # then perform an inner join operation to obtain the table with corresponding content ids.

    # we only need the channel and content ids
    channel_and_contents = topic_and_contents[["channel", "content_ids"]]

    global topic_trees_contents
    topic_trees_contents = dict()

    # loop through each row and aggregate them together
    for idx in channel_and_contents.index:
        # find the string value in the "content_ids" column
        content_ids = channel_and_contents.loc[idx, "content_ids"]
        # find the channel UID in the "channel" column
        channel = channel_and_contents.loc[idx, "channel"]
        # if channel not already in topic_trees_contents, initialize it
        if not channel in topic_trees_contents:
            topic_trees_contents[channel] = []
        # loop through the content_id and add it to the array
        for content_id in content_ids.split():
            topic_trees_contents[channel].append(content_id)

    # add empty arrays
    for channel in topic_trees_id_list:
        if channel not in topic_trees_contents:
            topic_trees_contents[channel] = []

initialize_topic_trees()
initialize_topic_tree_contents()