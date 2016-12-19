import sys
import math
import time
import datetime
import numpy as np
# import pandas as pd
import pandas
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster.k_means_ import KMeans
from sklearn import linear_model
from scipy.spatial import Voronoi


def get_distance(data1, data2):
    data3 = np.sum(np.power(data1 - data2, 2))
    data3 = np.sqrt(data3)
    return data3


def print_time_stamp(t2, t1):
    curr_time = datetime.datetime.fromtimestamp(t2).strftime('%Y-%m-%d %H:%M:%S')
    print("Current time -> %s " % curr_time)
    delta = math.floor(t2 - t1) + 1
    h = math.floor(delta / 3600)
    m = math.floor(delta - 3600 * h) / 60
    s = delta % 60
    print("Delta time -> %dh %dm %ds" % (h, m, s))
    print("Delta time in seconds -> %f" % delta)


def fit_and_predict(train_file, test_file, result_file, type):
    # load datesets as pandas dataframe
    train_sample = pd.read_csv(train_file)
    test_sample = pd.read_csv(test_file)
    
    X = train_sample[train_sample.columns[1:]].values
    y = train_sample[train_sample.columns[0]].values
    
    # fit classificator
    if type == "knn":
        clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        clf.fit(X, y)
    elif type == "linearsvc":
        clf = LinearSVC()
        X = np.array(X, dtype=float)
        X = scale(X)
        clf.fit(X, y)
    elif type == "logregress":
        # Logistic regression, despite its name,
        # is a linear model for classification
        # rather than regression
        clf = linear_model.LogisticRegression()
        X = np.array(X, dtype=float)
        X = scale(X)
        clf.fit(X, y)
    
    # save results
    T = test_sample.values
    result = open(result_file, "w")
    result.write('"ImageId","Label"\n')
    prediction = clf.predict(T)
    for i in range(len(prediction)):
        string = str(i + 1) + ',' + '"' + str(prediction[i]) + '"\n'
        result.write(string)
    
    result.close()
    

def main(argv):
    start_time = time.time()
    curr_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print("before reading -> %s " % curr_time)
    
    train_sample = pandas.read_csv(argv[1])
    test_sample = pandas.read_csv(argv[2])
    result_file = open(argv[3], "w")
    # type = argv[4]
    
    x_array = train_sample[train_sample.columns[1:]].values
    y = train_sample[train_sample.columns[0]].values

    curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("before training -> %s " % curr_time)
    
    clf = KMeans(init="random", n_jobs=-1)
    clf.fit(x_array, y)

    curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("before predicting -> %s " % curr_time)
    
    t = test_sample.values
    prediction = clf.predict(t)

    result_file.write('"ImageId","Label"\n')
    for i in range(len(prediction)):
        string = str(i + 1) + "," + str(prediction[i]) + '\n'
        result_file.write(string)
    result_file.close()
    
    print_time_stamp(time.time(), start_time)


if __name__ == "__main__":
    main(sys.argv)
    