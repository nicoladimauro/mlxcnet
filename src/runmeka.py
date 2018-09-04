#!/usr/bin/python3

from meka import Meka
import numpy as np
import argparse
import sklearn.metrics
from tabulate import tabulate

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
from dataset import Dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('-f', type=int, nargs='?',
                    default=5,
                    help='Number of folds for the dataset')


parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/meka/',
                    help='Output dir path')

parser.add_argument('-mc', type=str, nargs='?',
                    default="meka.classifiers.multilabel.LC",
                    help='Meka classifier')

parser.add_argument('-wc', type=str, nargs='?',
                    default="weka.classifiers.bayes.NaiveBayes",
#                    default="weka.classifiers.bayes.BayesNet -- -D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5",
                    help='Weka classifier')

parser.add_argument('-mp', type=str, nargs='?',
                    default='./meka/lib/',
                    help='Meka classpath')

parser.add_argument('-c', type=int, nargs=1,
                    help='Number of class labels.')


args = parser.parse_args()
(dataset_name,) = args.dataset

date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string
out_log_path = out_path + '/exp.log'

if not os.path.exists(os.path.dirname(out_log_path)):
    os.makedirs(os.path.dirname(out_log_path))


Accuracy = ['Accuracy']
Hamming_score = ['Hamming Score']
Exact_match = ['Exact match']
Time = ['Time']
Headers = ['Metric']

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.flush()


    for f in range(args.f):
        train_file_name = dataset_name + ".f" + str(f) + ".train.arff"
        test_file_name = dataset_name + ".f" + str(f) + ".test.arff"

        data = Dataset.load_arff("./data/"+test_file_name, args.c[0], endian = "big", input_feature_type = 'int', encode_nominal = True)

        meka = Meka(args.mc, args.wc, meka_classpath=args.mp)
        learn_start_t = perf_counter()
        predictions, statistics = meka.run("./data/"+train_file_name, "./data/" + test_file_name)
        learn_end_t = perf_counter()
        learning_time = (learn_end_t - learn_start_t)

        print("Accuracy :     :", statistics['Accuracy'])
        print('Hammingloss    :', statistics['Hammingloss'])
        print('Exactmatch', statistics['Exactmatch'])
        print('BuildTime', statistics['BuildTime'])
        print('TestTime', statistics['TestTime'])

        Accuracy.append(sklearn.metrics.jaccard_similarity_score(data['Y'], predictions))
        Hamming_score.append(1-sklearn.metrics.hamming_loss(data['Y'], predictions))
        Exact_match.append(1-sklearn.metrics.zero_one_loss(data['Y'], predictions))
        Time.append(learning_time)
        Headers.append("Fold "+ str(f))


    out_log.write(tabulate([Accuracy, Hamming_score, Exact_match, Time], 
                           headers=Headers, tablefmt='orgtbl'))

    out_log.write('\n\nAccuracy (mean/std)      : %f / %f' % (np.mean(np.array(Accuracy[1:])),np.std(np.array(Accuracy[1:]))))
    out_log.write('\nHamming score (mean/std)   : %f / %f' % (np.mean(np.array(Hamming_score[1:])), np.std(np.array(Hamming_score[1:]))))
    out_log.write('\nExact match (mean/std)     : %f / %f' % (np.mean(np.array(Exact_match[1:])), np.std(np.array(Exact_match[1:]))))
    out_log.write('\nTime (mean/std)            : %f / %f' % (np.mean(np.array(Time[1:])), np.std(np.array(Time[1:]))))
    out_log.flush()
    

print(tabulate([Accuracy, Hamming_score, Exact_match, Time], 
               headers=Headers, tablefmt='orgtbl'))


