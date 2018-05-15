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

parser.add_argument('-k', type=int, nargs='+',
                    default=[1],
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

with open(out_log_path, 'w') as out_log:



    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.flush()
    #
    # looping over all parameters combinations
    for alpha in alphas:
        for min_instances in m_instances:
            for min_features in m_features:

                Headers = ['Metric']

                Train_Accuracy_mpe = ['Train Accuracy mpe']
                Train_Hamming_score_mpe = ['Train Hamming Score mpe']
                Train_Exact_match_mpe = ['Train Exact match mpe']

                Test_Accuracy_mpe = ['Test Accuracy mpe']
                Test_Hamming_score_mpe = ['Test Hamming Score mpe']
                Test_Exact_match_mpe = ['Test Exact match mpe']

                Train_Accuracy_marg = ['Train Accuracy marg']
                Train_Hamming_score_marg = ['Train Hamming Score marg']
                Train_Exact_match_marg = ['Train Exact match marg']

                Test_Accuracy_marg = ['Test Accuracy marg']
                Test_Hamming_score_marg = ['Test Hamming Score marg']
                Test_Exact_match_marg = ['Test Exact match marg']

                Learning_time = ['Learning time']
                Testing_time = ['Testing time']



                for f in range(args.f):
                
                    C = None

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

                    learn_start_t = perf_counter()
                    C = mlcsn(train_data, 
                              min_instances=min_instances_, 
                              min_features=min_features, 
                              alpha=alpha, 
                              leaf_vars = l_vars,
                              n_labels = train['Y'].shape[1],
                              multilabel = True,
                              ml_tree_structure=args.ts,
                              xcnet=args.x)

                    C.fit()
                    learn_end_t = perf_counter()

                    learning_time = (learn_end_t - learn_start_t)


                    test_data = Dataset.load_arff("./data/"+dataset_name+".f"+str(f)+".test.arff", n_labels, endian = "big", input_feature_type = 'int', encode_nominal = True)
                    test_start_t = perf_counter()
                    Test_Y_pred_mpe = C.compute_predictions(test_data['X'], n_labels)
                    test_end_t = perf_counter()
                    testing_time = (test_end_t - test_start_t)

                    Test_Y_pred_marg = C.marginal_inference(test_data['X'], n_labels)

                    Train_Y_pred_mpe = C.compute_predictions(train['X'], n_labels)
                    Train_Y_pred_marg = C.marginal_inference(train['X'], n_labels)

                    Test_Accuracy_mpe.append(sklearn.metrics.jaccard_similarity_score(test_data['Y'], Test_Y_pred_mpe))
                    Test_Hamming_score_mpe.append(1-sklearn.metrics.hamming_loss(test_data['Y'], Test_Y_pred_mpe))
                    Test_Exact_match_mpe.append(1-sklearn.metrics.zero_one_loss(test_data['Y'], Test_Y_pred_mpe))
                    Test_Accuracy_marg.append(sklearn.metrics.jaccard_similarity_score(test_data['Y'], Test_Y_pred_marg))
                    Test_Hamming_score_marg.append(1-sklearn.metrics.hamming_loss(test_data['Y'], Test_Y_pred_marg))
                    Test_Exact_match_marg.append(1-sklearn.metrics.zero_one_loss(test_data['Y'], Test_Y_pred_marg))

                    Train_Accuracy_mpe.append(sklearn.metrics.jaccard_similarity_score(train['Y'], Train_Y_pred_mpe))
                    Train_Hamming_score_mpe.append(1-sklearn.metrics.hamming_loss(train['Y'], Train_Y_pred_mpe))
                    Train_Exact_match_mpe.append(1-sklearn.metrics.zero_one_loss(train['Y'], Train_Y_pred_mpe))
                    Train_Accuracy_marg.append(sklearn.metrics.jaccard_similarity_score(train['Y'], Train_Y_pred_marg))
                    Train_Hamming_score_marg.append(1-sklearn.metrics.hamming_loss(train['Y'], Train_Y_pred_marg))
                    Train_Exact_match_marg.append(1-sklearn.metrics.zero_one_loss(train['Y'], Train_Y_pred_marg))

                    Learning_time.append(learning_time)
                    Testing_time.append(testing_time)
                    Headers.append("Fold "+ str(f))

                  
                    train_probs = C.compute_probs(train)
                    test_probs = C.compute_probs(test_data)

                    print("Train:", np.mean(train_probs[0]),np.mean(train_probs[1]),np.mean(train_probs[2]),np.mean(train_probs[3]))
                    print("Test:", np.mean(test_probs[0]),np.mean(test_probs[1]),np.mean(test_probs[2]),np.mean(test_probs[3]))

                    print(tabulate([Train_Accuracy_mpe, Train_Accuracy_marg, Train_Hamming_score_mpe, Train_Hamming_score_marg, 
                                    Train_Exact_match_mpe, Train_Exact_match_marg, 
                                    Test_Accuracy_mpe, Test_Accuracy_marg, Test_Hamming_score_mpe, Test_Hamming_score_marg, 
                                    Test_Exact_match_mpe, Test_Exact_match_marg, 
                                    Learning_time, Testing_time], 
                                   headers=Headers, tablefmt='orgtbl'))

                print('\nTrain Accuracy mpe (mean/std)        :', np.mean(np.array(Train_Accuracy_mpe[1:])),"/",np.std(np.array(Train_Accuracy_mpe[1:])))
                print('Train Accuracy marg (mean/std)       :', np.mean(np.array(Train_Accuracy_marg[1:])),"/",np.std(np.array(Train_Accuracy_marg[1:])))
                print('Train Hamming score mpe (mean/std)   :', np.mean(np.array(Train_Hamming_score_mpe[1:])), "/", np.std(np.array(Train_Hamming_score_mpe[1:])))
                print('Train Hamming score marg (mean/std)  :', np.mean(np.array(Train_Hamming_score_marg[1:])), "/", np.std(np.array(Train_Hamming_score_marg[1:])))
                print('Train Exact match mpe (mean/std)     :', np.mean(np.array(Train_Exact_match_mpe[1:])), "/", np.std(np.array(Train_Exact_match_mpe[1:])))
                print('Train Exact match marg (mean/std)    :', np.mean(np.array(Train_Exact_match_marg[1:])), "/", np.std(np.array(Train_Exact_match_marg[1:])))

                print('\nTest Accuracy mpe (mean/std)        :', np.mean(np.array(Test_Accuracy_mpe[1:])),"/",np.std(np.array(Test_Accuracy_mpe[1:])))
                print('Test Accuracy marg (mean/std)       :', np.mean(np.array(Test_Accuracy_marg[1:])),"/",np.std(np.array(Test_Accuracy_marg[1:])))
                print('Test Hamming score mpe (mean/std)   :', np.mean(np.array(Test_Hamming_score_mpe[1:])), "/", np.std(np.array(Test_Hamming_score_mpe[1:])))
                print('Test Hamming score marg (mean/std)  :', np.mean(np.array(Test_Hamming_score_marg[1:])), "/", np.std(np.array(Test_Hamming_score_marg[1:])))
                print('Test Exact match mpe (mean/std)     :', np.mean(np.array(Test_Exact_match_mpe[1:])), "/", np.std(np.array(Test_Exact_match_mpe[1:])))
                print('Test Exact match marg (mean/std)    :', np.mean(np.array(Test_Exact_match_marg[1:])), "/", np.std(np.array(Test_Exact_match_marg[1:])))

                print('\nLearning Time (mean/std)            :', np.mean(np.array(Learning_time[1:])), "/", np.std(np.array(Learning_time[1:])))
                print('Testing Time (mean/std)             :', np.mean(np.array(Testing_time[1:])), "/", np.std(np.array(Testing_time[1:])))


                out_log.write(tabulate([Train_Accuracy_mpe, Train_Accuracy_marg, Train_Hamming_score_mpe, Train_Hamming_score_marg, 
                                    Train_Exact_match_mpe, Train_Exact_match_marg, 
                                    Test_Accuracy_mpe, Test_Accuracy_marg, Test_Hamming_score_mpe, Test_Hamming_score_marg, 
                                    Test_Exact_match_mpe, Test_Exact_match_marg, 
                                    Learning_time, Testing_time], 
                                   headers=Headers, tablefmt='orgtbl'))

                out_log.write('\nTrain Accuracy mpe (mean/std)        : %f / %f' % ( np.mean(np.array(Train_Accuracy_mpe[1:])),np.std(np.array(Train_Accuracy_mpe[1:]))))
                out_log.write('\nTrain Accuracy marg (mean/std)       : %f / %f' % ( np.mean(np.array(Train_Accuracy_marg[1:])),np.std(np.array(Train_Accuracy_marg[1:]))))
                out_log.write('\nTrain Hamming score mpe (mean/std)   : %f / %f' % ( np.mean(np.array(Train_Hamming_score_mpe[1:])), np.std(np.array(Train_Hamming_score_mpe[1:]))))
                out_log.write('\nTrain Hamming score marg (mean/std)  : %f / %f' % ( np.mean(np.array(Train_Hamming_score_marg[1:])), np.std(np.array(Train_Hamming_score_marg[1:]))))
                out_log.write('\nTrain Exact match mpe (mean/std)     : %f / %f' % ( np.mean(np.array(Train_Exact_match_mpe[1:])), np.std(np.array(Train_Exact_match_mpe[1:]))))
                out_log.write('\nTrain Exact match marg (mean/std)    : %f / %f' % ( np.mean(np.array(Train_Exact_match_marg[1:])), np.std(np.array(Train_Exact_match_marg[1:]))))

                out_log.write('\nTest Accuracy mpe (mean/std)        : %f / %f' % ( np.mean(np.array(Test_Accuracy_mpe[1:])),np.std(np.array(Test_Accuracy_mpe[1:]))))
                out_log.write('\nTest Accuracy marg (mean/std)       : %f / %f' % ( np.mean(np.array(Test_Accuracy_marg[1:])),np.std(np.array(Test_Accuracy_marg[1:]))))
                out_log.write('\nTest Hamming score mpe (mean/std)   : %f / %f' % ( np.mean(np.array(Test_Hamming_score_mpe[1:])), np.std(np.array(Test_Hamming_score_mpe[1:]))))
                out_log.write('\nTest Hamming score marg (mean/std)  : %f / %f' % ( np.mean(np.array(Test_Hamming_score_marg[1:])), np.std(np.array(Test_Hamming_score_marg[1:]))))
                out_log.write('\nTest Exact match mpe (mean/std)     : %f / %f' % ( np.mean(np.array(Test_Exact_match_mpe[1:])), np.std(np.array(Test_Exact_match_mpe[1:]))))
                out_log.write('\nTest Exact match marg (mean/std)    : %f / %f' % ( np.mean(np.array(Test_Exact_match_marg[1:])), np.std(np.array(Test_Exact_match_marg[1:]))))

                out_log.write('\nLearning Time (mean/std)          : %f / %f' % ( np.mean(np.array(Learning_time[1:])), np.std(np.array(Learning_time[1:]))))
                out_log.write('\nTesting Time (mean/std)          : %f / %f' % ( np.mean(np.array(Testing_time[1:])), np.std(np.array(Testing_time[1:]))))



                out_log.flush()
