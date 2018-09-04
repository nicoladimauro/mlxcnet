"""
MIT License

Copyright (c) 2018 Nicola Di Mauro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from mlcsn import mlcsn
import numpy as np
import argparse
import shutil
import sklearn.metrics

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
import os
import logging
import random
from dataset import Dataset
import arff
from tabulate import tabulate

def stats_format(stats_list, separator, digits=5):
    formatted = []
    float_format = '{0:.' + str(digits) + 'f}'
    for stat in stats_list:
        if isinstance(stat, int):
            formatted.append(str(stat))
        elif isinstance(stat, float):
            formatted.append(float_format.format(stat))
        else:
            formatted.append(stat)
    # concatenation
    return separator.join(formatted)



#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. flags)')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/mlcsn/',
                    help='Output dir path')

parser.add_argument('-x', action='store_true', default=False,
                    help='Extremely Randomized CNets.')

parser.add_argument('-k', type=int, nargs='?',
                    default=1,
                    help='Number of components to use. If greater than 1, then a bagging approach is used.')

parser.add_argument('-d', type=float, nargs='+',
                    default=[10],
                    help='Min number of instances in a slice to split.')

parser.add_argument('-s', type=int, nargs='+',
                    default=[4],
                    help='Min number of features in a slice to split.')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[1.0],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('-c', type=int, nargs=1,
                    help='Number of class labels.')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('-l',  action='store_true', default=True,
                    help='Labels as leafs.')

parser.add_argument('-ts', type=int, nargs='?',
                    default=2,
                    help='Tree structure')

parser.add_argument('-f', type=int, nargs='?',
                    default=5,
                    help='Number of folds for the dataset')



#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# gathering parameters
alphas = args.alpha
n_components = args.k
m_instances = args.d
m_features = args.s

n_labels = args.c[0]

#
# elaborating the dataset
#
logging.info('Loading datasets: %s', args.dataset)
(dataset_name,) = args.dataset

#
# Opening the file for test prediction
#
logging.info('Opening log file...')
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string
out_log_path = out_path + '/exp.log'

#
# creating dir if non-existant
if not os.path.exists(os.path.dirname(out_log_path)):
    os.makedirs(os.path.dirname(out_log_path))

def create_bags(X, max_components):
    bags = [None] * max_components
    if max_components == 1:
        bags[0] = X
    else:
        for i in range(max_components):
            bags[i] = create_bag(X)
    return bags

def create_bag(X):
    n_instances = X.shape[0]
    bag = np.zeros((n_instances, X.shape[1]), dtype='int')
    for i in range(n_instances):
        choice = random.randint(0, X.shape[0]-1)
        bag[i] = X[choice]
    return bag

with open(out_log_path, 'w') as out_log:



    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.flush()
    #
    # looping over all parameters combinations
    for alpha in alphas:
        for min_instances in m_instances:
            for min_features in m_features:

                Headers = ['Metric']

                Test_Accuracy_mpe = ['Test Accuracy mpe']
                Test_Hamming_score_mpe = ['Test Hamming Score mpe']
                Test_Exact_match_mpe = ['Test Exact match mpe']

                Test_Accuracy_marg = ['Test Accuracy marg']
                Test_Hamming_score_marg = ['Test Hamming Score marg']
                Test_Exact_match_marg = ['Test Exact match marg']

                Learning_time = ['Learning time']
                Testing_time_mpe = ['Testing time mpe']
                Testing_time_marg = ['Testing time marg']                


                for f in range(args.f):
                
                    # initing the random generators
                    seed = args.seed
                    numpy_rand_gen = np.random.RandomState(seed)
                    random.seed(seed)

                    train = Dataset.load_arff("./data/"+dataset_name+".f"+str(f)+".train.arff", n_labels, endian = "big", input_feature_type = 'int', encode_nominal = True)
                    train_data = np.concatenate((train['X'],train['Y']), axis = 1)

                    if args.l:
                        l_vars = [i+train['X'].shape[1] for i in range(train['Y'].shape[1])]
                    else:
                        l_vars = []

                    min_instances_ = min_instances
                    if min_instances <= 1:
                        min_instances_ = int(train['X'].shape[0] * min_instances)+1
                        print("Setting min_instances to ", min_instances_)
                    else:
                        min_instances_ = min_instances

                    test_data = Dataset.load_arff("./data/"+dataset_name+".f"+str(f)+".test.arff", n_labels, endian = "big", input_feature_type = 'int', encode_nominal = True)
                    
                    test_predictions_mpe = np.zeros((test_data['Y'].shape[0], test_data['Y'].shape[1]),dtype=np.int)
                    test_predictions_marg = np.zeros((test_data['Y'].shape[0], test_data['Y'].shape[1]),dtype=np.int)                    
                    bag_predictions_mpe = np.zeros((test_data['Y'].shape[0], test_data['Y'].shape[1]),dtype=np.int)
                    bag_predictions_marg = np.zeros((test_data['Y'].shape[0], test_data['Y'].shape[1]),dtype=np.float)

                    testing_time_mpe = 0
                    testing_time_marg = 0
                    learning_time = 0
                    
                    for i in range(n_components):
                        #training_bag = create_bag(train_data)
                        training_bag = train_data

                        start_t = perf_counter()
                        C = mlcsn(training_bag, 
                                  min_instances=min_instances_, 
                                  min_features=min_features, 
                                  alpha=alpha, 
                                  leaf_vars = l_vars,
                                  n_labels = train['Y'].shape[1],
                                  multilabel = True,
                                  ml_tree_structure=args.ts,
                                  xcnet=args.x)
                        C.fit()
                        end_t = perf_counter()
                        learning_time +=  (end_t - start_t)

                        start_t = perf_counter()
                        bag_predictions_mpe = bag_predictions_mpe + C.compute_predictions(test_data['X'], n_labels)
                        end_t = perf_counter()
                        testing_time_mpe += (end_t - start_t)
                        start_t = perf_counter()
                        bag_predictions_marg = bag_predictions_marg + C.marginal_inference1(test_data['X'], n_labels)
                        end_t = perf_counter()
                        testing_time_marg += (end_t - start_t)

                    
                    for i in range(bag_predictions_mpe.shape[0]):
                        for l in range(n_labels):
                            if bag_predictions_mpe[i,l] > n_components/2:
                                test_predictions_mpe[i,l] = 1
                            if bag_predictions_marg[i,l] > n_components/2:
                                test_predictions_marg[i,l] = 1
                    
                    
                   
                    Test_Accuracy_mpe.append(sklearn.metrics.jaccard_similarity_score(test_data['Y'], test_predictions_mpe))
                    Test_Hamming_score_mpe.append(1-sklearn.metrics.hamming_loss(test_data['Y'], test_predictions_mpe))
                    Test_Exact_match_mpe.append(1-sklearn.metrics.zero_one_loss(test_data['Y'], test_predictions_mpe))
                    Test_Accuracy_marg.append(sklearn.metrics.jaccard_similarity_score(test_data['Y'], test_predictions_marg))
                    Test_Hamming_score_marg.append(1-sklearn.metrics.hamming_loss(test_data['Y'], test_predictions_marg))
                    Test_Exact_match_marg.append(1-sklearn.metrics.zero_one_loss(test_data['Y'], test_predictions_marg))

                    Learning_time.append(learning_time)
                    Testing_time_mpe.append(testing_time_mpe)
                    Testing_time_marg.append(testing_time_marg)                    
                    Headers.append("Fold "+ str(f))

                  
                    print(tabulate([Test_Accuracy_mpe, Test_Accuracy_marg, Test_Hamming_score_mpe, Test_Hamming_score_marg,
                                    Test_Exact_match_mpe, Test_Exact_match_marg, 
                                    Learning_time, Testing_time_mpe, Testing_time_mpe],
                                   headers=Headers, tablefmt='orgtbl'))

                print('\nTest Accuracy mpe (mean/std)        :', np.mean(np.array(Test_Accuracy_mpe[1:])),"/",np.std(np.array(Test_Accuracy_mpe[1:])))
                print('Test Accuracy marg (mean/std)       :', np.mean(np.array(Test_Accuracy_marg[1:])),"/",np.std(np.array(Test_Accuracy_marg[1:])))
                print('Test Hamming score mpe (mean/std)   :', np.mean(np.array(Test_Hamming_score_mpe[1:])), "/", np.std(np.array(Test_Hamming_score_mpe[1:])))
                print('Test Hamming score marg (mean/std)  :', np.mean(np.array(Test_Hamming_score_marg[1:])), "/", np.std(np.array(Test_Hamming_score_marg[1:])))
                print('Test Exact match mpe (mean/std)     :', np.mean(np.array(Test_Exact_match_mpe[1:])), "/", np.std(np.array(Test_Exact_match_mpe[1:])))
                print('Test Exact match marg (mean/std)    :', np.mean(np.array(Test_Exact_match_marg[1:])), "/", np.std(np.array(Test_Exact_match_marg[1:])))

                print('\nLearning Time (mean/std)            :', np.mean(np.array(Learning_time[1:])), "/", np.std(np.array(Learning_time[1:])))
                print('Testing Time mpe (mean/std)             :', np.mean(np.array(Testing_time_mpe[1:])), "/", np.std(np.array(Testing_time_mpe[1:])))
                print('Testing Time marg (mean/std)             :', np.mean(np.array(Testing_time_marg[1:])), "/", np.std(np.array(Testing_time_marg[1:])))                

                out_log.write(tabulate([Test_Accuracy_mpe, Test_Accuracy_marg, Test_Hamming_score_mpe, Test_Hamming_score_marg, 
                                        Test_Exact_match_mpe, Test_Exact_match_marg, 
                                        Learning_time, Testing_time_mpe, Testing_time_marg], 
                                       headers=Headers, tablefmt='orgtbl'))

                out_log.write('\nTest Accuracy mpe (mean/std)        : %f / %f' % ( np.mean(np.array(Test_Accuracy_mpe[1:])),np.std(np.array(Test_Accuracy_mpe[1:]))))
                out_log.write('\nTest Accuracy marg (mean/std)       : %f / %f' % ( np.mean(np.array(Test_Accuracy_marg[1:])),np.std(np.array(Test_Accuracy_marg[1:]))))
                out_log.write('\nTest Hamming score mpe (mean/std)   : %f / %f' % ( np.mean(np.array(Test_Hamming_score_mpe[1:])), np.std(np.array(Test_Hamming_score_mpe[1:]))))
                out_log.write('\nTest Hamming score marg (mean/std)  : %f / %f' % ( np.mean(np.array(Test_Hamming_score_marg[1:])), np.std(np.array(Test_Hamming_score_marg[1:]))))
                out_log.write('\nTest Exact match mpe (mean/std)     : %f / %f' % ( np.mean(np.array(Test_Exact_match_mpe[1:])), np.std(np.array(Test_Exact_match_mpe[1:]))))
                out_log.write('\nTest Exact match marg (mean/std)    : %f / %f' % ( np.mean(np.array(Test_Exact_match_marg[1:])), np.std(np.array(Test_Exact_match_marg[1:]))))

                out_log.write('\nLearning Time (mean/std)          : %f / %f' % ( np.mean(np.array(Learning_time[1:])), np.std(np.array(Learning_time[1:]))))
                out_log.write('\nTesting Time mpe (mean/std)          : %f / %f' % ( np.mean(np.array(Testing_time_mpe[1:])), np.std(np.array(Testing_time_marg[1:]))))
                out_log.write('\nTesting Time marg (mean/std)          : %f / %f' % ( np.mean(np.array(Testing_time_marg[1:])), np.std(np.array(Testing_time_marg[1:]))))                



                out_log.flush()
