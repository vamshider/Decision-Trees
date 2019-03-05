# decision_tree.py
# ----------------

import numpy as np
import matplotlib.pyplot as mtplt

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    part = {v: (x == v).nonzero()[0] for v in np.unique(x)}
    return part


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    list_of_classes, count_of_classes = np.unique(y,return_counts = True)
    entropy = 0
    
    for i in range(len(count_of_classes)):
        p = count_of_classes[i]/len(y)
        entropy -= p*np.log2(p)
        
    return entropy


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    x_values, x_counts = np.unique(x, return_counts=True)
    p = x_counts / float(len(x))                

    I = entropy(y)
    for p, v in zip(p, x_values):
        I -= p*entropy(y[x == v])

    return I


def id3(x, y, max_depth, attribute_value_pairs=None, depth=0):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    list_of_classes, count_per_class_list = np.unique(y, return_counts=True)

    #1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
    if len(list_of_classes) == 1:
        return list_of_classes[0]
    
    #2. If the set of attribute-value pairs is empty or if the max_depth is reached, then return the most common value of y
    if len(np.array(range(x.shape[1]))) == 0 or depth == max_depth:
        return list_of_classes[np.argmax(count_per_class_list)]
    
    

    # If split_criteria=None, then create a Cartesian product of splits
    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, v) for v in np.unique(x[:, i])] for i in range(x.shape[1])])

    #Calculating the best attribute-value pair using the mutual information
    information_gain = []
    for i, j in attribute_value_pairs:
        information_gain.append(mutual_information(np.array(x[:, i] == j).astype(int), y))
    

    #The best attribute-value pair
    (attribute, value) = attribute_value_pairs[np.argmax(information_gain)]

    partitions = partition(list(x[:, attribute] == value))
    
    # Removing the attribute-value pair which formed a node in the Decision tree from the list of attributes
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(np.all(attribute_value_pairs == (attribute, value))), 0)

    #Creating a node for the found best attribute-value pair
    root = {}
    
    for split_value, indices in partitions.items():
        x_subset = x.take(indices, axis=0)
        y_subset = y.take(indices, axis=0)
        decision = bool(split_value)

        root[(attribute, value, decision)] = id3(x_subset, y_subset, max_depth, attribute_value_pairs, depth + 1)

    return root


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    for criteria, subtree in tree.items():
        index = criteria[0]
        value = criteria[1]
        decision = criteria[2]

        if decision == (x[index] == value):
            if isinstance(subtree, dict):
                label = predict_example(x, subtree)
            else:
                label = subtree

            return label
        
    
def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    return (1/y_true.size)*(np.sum(y_true != y_pred))


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusion_matrix(true_label, predicted_label):

    labels = np.unique(true_label)

    matrix = [[0 for x in range(len(labels))] for y in range(len(labels))]
    for t, p in zip(true_label, predicted_label):
        #matrix[t][p] += 1
        if t == 1 and p == 1:
            matrix[0][0] += 1
        elif t == 0 and p == 0:
            matrix[1][1] += 1
        elif t == 1 and p == 0:
            matrix[0][1] += 1
        elif t == 0 and p == 1:
            matrix[1][0] +=1
            
    return matrix
    
    
if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    train_error = []
    test_error = []
    # Learn a decision tree for the depths 1 to 10 
    for d in range(1, 11):
        
        decision_tree = id3(Xtrn, ytrn, d)
        y_trpred = [predict_example(x, decision_tree) for x in Xtrn]
        train_error.append(compute_error(ytrn, y_trpred))
        
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        test_error.append(compute_error(ytst, y_pred))
		
        if(d == 2):
            dtree_dpth_2 = decision_tree
            ypred_dpth_2 = y_pred.copy()
        if(d == 3):
            dtree_dpth_3 = decision_tree
            ypred_dpth_3 = y_pred.copy()


    #Calculating the Average Train and Test error for the Monks-1 Dataset
    average_train_error = sum(train_error)/len(train_error)
    average_test_error = sum(test_error)/len(test_error)
    
    print('Average Train Error for MONKS-1 Dataset = {0:4.2f}%.'.format(average_train_error * 100))
    print('Average Test Error for MONKS-1 Dataset = {0:4.2f}%.'.format(average_test_error * 100))

    # Plotting the Training and Test Errors
    mtplt.title("Training and Testing Error Curves for MONKS-1 Dataset")
    mtplt.xlabel("Depth")
    mtplt.ylabel("Error")
    mtplt.plot(range(1, 11), train_error, '--',  label="Training Error")
    mtplt.plot(range(1, 11), test_error, label="Testing Error")
    mtplt.legend()
    mtplt.show()
    
    
    #Decision Tree and Confusion Matrix on dataset monks-1 for the depths 2 and 3
	#Decision Tree and Confusion Matrix on dataset monks-1 for depth = 1 and 2
    visualize(dtree_dpth_2)
    cm = confusion_matrix(ytst, ypred_dpth_2)
    print(cm)
    print('\n')
    visualize(dtree_dpth_3)
    cm = confusion_matrix(ytst, ypred_dpth_3)
    print(cm)
    
    #----------------------------------------------------------------------------------------------------------