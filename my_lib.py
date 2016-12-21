import random
import sys
import math
import time
import datetime
import numpy as np
# import pandas as pd
import pandas
from scipy.spatial.distance import euclidean
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster.k_means_ import KMeans
from sklearn import linear_model
from scipy.spatial import Voronoi
import copy


def get_distance(a1_list, a2_list):
    # return np.sqrt(np.sum(np.power(a1 - a2, 2)))  # sqrt( (x11 - x21)^2 + ... + (x1n - x2n)^2 )
    return max([euclidean(a1, a2) for a1, a2 in zip(a1_list, a2_list)])


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
    
    
def train_k_means_by_step(n_clusters, init_cluster_centers, x_array, eps):
    # eps = 1e-4
    # eps = 0.1
    # eps = 100.0
    # prev_sample = np.array(clf.cluster_centers_, np.float)
    prev_centers = init_cluster_centers
    clf = KMeans(init=prev_centers, n_clusters=n_clusters, n_init=1, n_jobs=-1, tol=eps, max_iter=1)
    # if isinstance(prev_centers, str):
    #     prev_centers = clf.cluster_centers_
    clf.fit(x_array)
    new_centers = clf.cluster_centers_
    
    centers_list = [prev_centers, new_centers]
    args = [1]
    values = [clf.inertia_]
    while get_distance(prev_centers, new_centers) > eps:
        prev_centers = new_centers
        clf = KMeans(init=prev_centers, n_clusters=n_clusters, n_init=1, n_jobs=-1, tol=eps, max_iter=1).fit(x_array)
        new_centers = clf.cluster_centers_
        args.append(len(args) + 1)
        values.append(clf.inertia_)
        centers_list.append(new_centers)
    # print "k = %s, len centers = %s" % (n_clusters, len(f_values))
    return args, values, centers_list


def get_random_centers(x_array, n_clusters):
    return np.array([random.choice(x_array) for i in range(n_clusters)])


def get_k_away_centers(x_array, n_cluster):
    away_centers = [random.choice(x_array)]
    for i in range(n_cluster - 1):
        distances = [
            reduce(lambda d, y: d + euclidean(x, y), away_centers, 0.0)
            for x in x_array
        ]
        index = distances.index(max(distances))
        away_centers.append(x_array[index])
    return np.array(away_centers)


def train_k_means(n_clusters, init_type, x_array, y, eps, n_init):
    inertias = []
    iterations = []
    entropys = []
    for i in range(n_init):
        # fill matrix by zero
        n_matrix = np.zeros((len(y), n_clusters), dtype=np.int)
        if init_type == "random":
            init = "random"
        elif init_type == "k_away":
            init = get_k_away_centers(x_array, n_clusters)
        else:
            raise NotImplementedError
        
        clf = KMeans(init=init, n_clusters=n_clusters, n_init=1, n_jobs=-1, tol=eps)
        clf.fit(x_array)
        # Q value
        inertias.append(clf.inertia_)
        # iterations number
        iterations.append(clf.n_iter_)
        # labels
        for j in range(len(y)):
            digit = y[j]
            cluster = clf.labels_[j]
            n_matrix[digit][cluster] += 1
        n = float(len(y))

        # print "n_matrix = ", n_matrix
        Hyz = - reduce(lambda s, p: s + (p * math.log(p, 2) if p > 0 else 0),
                       [
                           n_matrix[digit][cluster] / n
                           for cluster in range(n_clusters)
                           for digit in range(len(y))
                       ],
                       0.0)
        Hz = - reduce(lambda s, p: s + (p * math.log(p, 2) if p > 0 else 0),
                      [
                          sum(n_matrix[digit], 0.0) / n
                          for digit in range(len(y))
                          ],
                      0.0)
        entropys.append(Hyz - Hz)
        
    return iterations, inertias, entropys


def main(argv):
    start_time = time.time()
    curr_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print("before reading -> %s " % curr_time)
    
    train_sample = pandas.read_csv(argv[1])
    test_sample = pandas.read_csv(argv[2])
    result_file = open(argv[3], "w")
    # type = argv[4]
    print "log = ", 5 * math.log(4,2)
    x_array = train_sample[train_sample.columns[1:]].values
    y = train_sample[train_sample.columns[0]].values
    curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("before training -> %s " % curr_time)
    
    # k = 1
    # init_centers = np.array([np.random.random_integers(0, 255, len(x_array[i]))
    #                          for i in range(k)])
    for k in range(1, 16):
        # init_centers = get_random_centers(x_array, k)
        # random_centers = get_random_centers(x_array, k)
        # init_centers = get_k_away_centers(x_array, k)
        train_k_means(k, "random", x_array, y, 10.0, 1)
        print "k = %s completed" % k



if __name__ == "__main__":
    main(sys.argv)
