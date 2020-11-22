from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import datasets
from helpers.data_loader import load, get_features_labels
from helpers.plots import roc_curves_plot
from algorithms.abc import ArtificialBeeColony
from numpy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import math

def run_cart(path, test_size):
    # df = load(path)
    # features, labels = get_features_labels(df)
    # X_train, X_test, y_train, y_test = get_train_test_data(features, labels, test_size)
    # y_test_binarized = class_binarize(y_test, [1, 2])

    # clf = DecisionTreeClassifier()

    newsgroups_train = datasets.fetch_20newsgroups(subset='train')
    newsgroups_test = datasets.fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', DecisionTreeClassifier()),
                        ])

    clf = train_cart(clf, X_train, y_train)

    # n_classes = y_test_binarized.shape[1]
    # y_pred_binarized = class_binarize(test_cart(clf, X_test), [1, 2])
    y_pred = test_cart(clf, X_test)
    # roc_auc_values, fpr, tpr = roc_auc(n_classes, y_test_binarized, y_pred_binarized)
    # roc_curves_plot(roc_auc_values, fpr, tpr, n_classes)

    print(reports(y_test, y_pred))

def run_cart_abc(path, t_s):
    # newsgroups_train = fetch_20newsgroups(subset='train')
    # newsgroups_test = fetch_20newsgroups(subset='test')
    # X_train = newsgroups_train.data
    # X_test = newsgroups_test.data
    # y_train = newsgroups_train.target
    # y_test = newsgroups_test.target
    # clf = Pipeline([('vect', CountVectorizer()),
    #                     ('tfidf', TfidfTransformer()),
    #                     ('clf', DecisionTreeClassifier()),
    #                     ])
    # clf.fit(X_train, y_train)
    # score = get_score(clf, X_test, y_test)
    # modification_rate = 0.3
    # cycle = 10
    # columns = [c for c in range(len(X_train))]
    # abc = ArtificialBeeColony(clf, columns, X_train, X_test, y_train, y_test, modification_rate)
    # accuracy, selected_features, bee, _ = abc.execute(cycle, score)
    # print(reports(y_test, bee.get_y_pred()))

    test_size = 0.3
    x_axis = []
    y_axis = []

    df = load(path)
    features, labels = get_features_labels(df)
    for i in range(1):
        # print(df)
        x_axis.append(test_size)
        X_train, X_test, y_train, y_test = get_train_test_data(features, labels, test_size)
        # y_test_binarized = class_binarize(y_test, [1, 2])
        clf = DecisionTreeClassifier()
        clf = train_cart(clf, X_train, y_train)
        # n_classes = y_test_binarized.shape[1]
        y_pred = test_cart(clf, X_test)
        # y_pred_binarized = class_binarize(y_pred, [1, 2])
        score = get_score(clf, X_test, y_test)
        modification_rate = 0.3
        max_limit = 3
        cycle = 2
        food_num = int(len(X_train) / 10)
        abc = ArtificialBeeColony(clf, X_train.columns, X_train, X_test, y_train, y_test, modification_rate, features.iloc[test], food_num)
        accuracy, selected_features, bee, _ = abc.execute(cycle, score, max_limit)
        # print(score)
        # print(clf.classes_)
        # print(accuracy)
        y_axis.append(accuracy)
        # print(f'Test Size: {test_size}')
        print(reports(y_test, bee.get_y_pred()))
        # test_size += 0.05
    # plt.plot(x_axis, y_axis)
    # plt.ylabel('accuracy')
    # plt.xlabel('test_size')
    # plt.axis([0, 1, 0.6, 0.9])
    # plt.show()

def run_cart_kfold(path, n_splits):
    df = load(path)
    features, labels = get_features_labels(df)
    cv = StratifiedKFold(n_splits = n_splits)

    avg_scores = np.array([])

    clf = DecisionTreeClassifier()

    x_axis = []
    y_axis = []
    y_axis_1 = []
    train_num = 900

    food_num_coef = 0

    while food_num_coef < 0.1:

        cycle = 2
        modification_rate = 0.85
        food_num_coef += 0.01
        max_limit = 3
        food_num = 100

        for i, (train, test) in enumerate(cv.split(features, labels)):
            probas_ = clf.fit(features.iloc[train], labels.iloc[train]).predict_proba(features.iloc[test])

            score = clf.score(features.iloc[test], labels.iloc[test])

            abc = ArtificialBeeColony(clf, features.columns, features.iloc[train], features.iloc[test], labels.iloc[train], labels.iloc[test], modification_rate, food_num)
            accuracy, selected_features, bee, _ = abc.execute(cycle, score, max_limit)

            print(f'Score Fold-{i+1}: {accuracy}')

            avg_scores = np.append(avg_scores, accuracy)
            
        avg = np.average(avg_scores)
        std = np.std(avg_scores)
        y_axis.append(avg)
        y_axis_1.append(std)
        x_axis.append(food_num_coef)
        print(f'Score Average-: {avg}')
    
    plt.plot(x_axis, y_axis)
    plt.ylabel('accuracy')
    plt.xlabel('food_num_coef')
    plt.show()

    plt.plot(x_axis, y_axis_1)
    plt.ylabel('standard deviation')
    plt.xlabel('food_num_coef')
    plt.show()

@ignore_warnings(category=ConvergenceWarning)
def run_comp_test(df):
    features, labels = get_features_labels(df)
    cv = StratifiedKFold(n_splits = 10)

    result = pd.DataFrame({},
                   columns=['DT', 'KNN', 'RF', "SVM", "ABC"])

    clf = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    svc = LinearSVC()

    cycle = 2
    modification_rate = 0.85
    max_limit = 4
    food_num = 100

    for i, (train, test) in enumerate(cv.split(features, labels)):
        predict_dt = clf.fit(features.iloc[train], labels.iloc[train]).predict(features.iloc[test])
        score_dt = clf.score(features.iloc[test], labels.iloc[test])

        abc = ArtificialBeeColony(clf, features.columns, features.iloc[train], features.iloc[test], labels.iloc[train], labels.iloc[test], modification_rate, food_num)
        accuracy, selected_features, bee, _ = abc.execute(cycle, 1, max_limit)

        predict_knn = knn.fit(features.iloc[train], labels.iloc[train]).predict(features.iloc[test])
        score_knn = knn.score(features.iloc[test], labels.iloc[test])

        predict_rf = rf.fit(features.iloc[train], labels.iloc[train]).predict(features.iloc[test])
        score_rf = rf.score(features.iloc[test], labels.iloc[test])

        predict_svm = svc.fit(features.iloc[train], labels.iloc[train]).predict(features.iloc[test])
        score_svc = svc.score(features.iloc[test], labels.iloc[test])

        row = pd.Series({'DT':score_dt,'KNN':score_knn,'RF':score_rf,'SVM':score_svc,'ABC':accuracy ** .5},name="Fold-"+str(1+i))
        result = result.append(row)

        # print("DT Fold-"+str(1+i))
        # print(reports(labels.iloc[test], predict_dt))
        # print("KNN Fold-"+str(1+i))
        # print(reports(labels.iloc[test], predict_knn))
        # print("RF Fold-"+str(1+i))
        # print(reports(labels.iloc[test], predict_rf))
        # print("SVM Fold-"+str(1+i))
        # print(reports(labels.iloc[test], predict_svm))
        # print("ABC Fold-"+str(1+i))
        # print(reports(labels.iloc[test], bee.get_y_pred()))

        print(f'Score Fold-{i+1}: {score_dt} {score_knn} {score_rf} {score_svc}  {accuracy ** .5}')

        # avg_scores = np.append(avg_scores, accuracy)
        
    # avg = np.average(avg_scores)
    mean = result.mean(axis = 0)
    mean.name = "Average"
    std = result.std(axis = 0)
    std.name = "Std"
    result = result.append(std)
    result = result.append(mean)
    print(result)


def run_compare(path):
    df = load(path)
    run_comp_test(df)

def run_iris_compare():
    data = datasets.load_iris()
    df = pd.DataFrame(data["data"])
    df['target'] = data.target
    run_comp_test(df)

def run_digits_compare():
    data = datasets.load_digits()
    df = pd.DataFrame(data["data"])
    df['target'] = data.target
    run_comp_test(df)

def run_wine_compare():
    data = datasets.load_wine()
    df = pd.DataFrame(data["data"])
    df['target'] = data.target
    run_comp_test(df)

def run_breast_cancer_compare():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data["data"])
    df['target'] = data.target
    run_comp_test(df)

def get_train_test_data(features, classes, test_size, random_state = 0):
    return train_test_split(features, classes, test_size = test_size, random_state = random_state)

def train_cart(cart, features, classes):
    return cart.fit(features, classes)

def test_cart(cart, x_test):
    return cart.predict(x_test)

def get_score(clf, x_test, y_true):
    return clf.score(x_test, y_true)

def class_binarize(y, classes):
    return label_binarize(y, classes = classes)

def roc_auc(n_classes, y_test, y_score):
    fpr = dict() # false positive rate
    tpr = dict() # true positive rate
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    return roc_auc, fpr, tpr

def reports(y_true, y_pred):
    return metrics.classification_report(y_true, y_pred)