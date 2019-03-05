#decision_tree_sklearn.py
#------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz as gv
from os import system

from sklearn import tree as dt 
from sklearn import metrics as mt
from sklearn.metrics import accuracy_score

#Predicts the classification label for a single example x using tree
#Returns the predicted label of x according to tree
def predict(x, tree):
    for split_criterion, sub_trees in tree.items():
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_trees) is dict:
                label = predict(x, sub_trees)
            else:
                label = sub_trees

            return label


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

def sklearn_evaluate(dataset_name="dummy name"):
    data_path = './'
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.train'),
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_train = data_Matrix[:, 0]
    X_train = data_Matrix[:, 1:]
    
    # Load the test data
    data_Matrix = np.genfromtxt((data_path + dataset_name + '.test'), 
                      missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y_test = data_Matrix[:, 0]
    X_test = data_Matrix[:, 1:]
    
    sk_train_error = []
    sk_test_error = []
    sk_y_test_pred = 0
    sk_tree_clf = {}

    for depth in range(1, 11):
        # train decision tree
        sk_tree_clf = dt.DecisionTreeClassifier(criterion='entropy', max_depth = depth)
        sk_tree_clf = sk_tree_clf.fit(X_train, y_train)
        
        y_train_pred = sk_tree_clf.predict(X_train)
        # train accuracy
        sk_train_error.append(1.000 - accuracy_score(y_train, y_train_pred))
        
        # prediction over test data
        sk_y_test_pred = sk_tree_clf.predict(X_test)
        # test acuuracy
        sk_test_error.append(1.000 - accuracy_score(y_test, sk_y_test_pred))
    
    # SKLEARN visualization (Check for a text file in your directory)
    with open("sk_classifier for " + dataset_name + ".txt", "w") as f:
        f = dt.export_graphviz(sk_tree_clf, out_file=f)

    print("\n SKLEARN: "+ dataset_name + " DEPTH vs ERROR PLOT")
    plt.title("SKLEARN: " + dataset_name + " Dataset")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.plot(range(1, 11), sk_train_error, '--',  label="Training Error")
    plt.plot(range(1, 11), sk_test_error, label="Testing Error")
    plt.legend()
    plt.show()
    
    # SKLEARN confusion matrix
    print("\n SKLEARN: " + dataset_name + " CONFUSION MATRIX")
    df = pd.DataFrame(
        mt.confusion_matrix(y_test, sk_y_test_pred),
        columns=['Predicted TRUE', 'Predicted FALSE'],
        index=['Actual TRUE', 'Actual FALSE']
    )  
    print(df)

if __name__ == '__main__':
    sklearn_evaluate("monks-1")
    sklearn_evaluate("monks-2")
    sklearn_evaluate("monks-3")
