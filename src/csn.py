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
import numpy as np
from scipy import sparse
import math 
import logging
import sys
import itertools 
import random
import sklearn.mixture

from nodes import Node, OrNode, TreeNode, is_or_node, is_tree_node
from cltree import Cltree

from scipy import optimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from time import perf_counter, process_time

###############################################################################

class Csn:

    _id_node_counter = 1
    _or_nodes = 0
    _leaf_nodes = 0
    _or_edges = 0
    _clt_edges = 0
    _cltrees = 0
    _depth = 0
    _mean_depth = 0

    @classmethod
    def init_stats(cls):
        Csn._id_node_counter = 1
        Csn._or_nodes = 0
        Csn._leaf_nodes = 0
        Csn._or_edges = 0
        Csn._clt_edges = 0
        Csn._cltrees = 0
        Csn._depth = 0
        Csn._mean_depth = 0
    
    def __init__(self, data, clt = None, ll = 0.0,  min_instances = 5, min_features = 3, 
                 alpha = 1.0, n_original_samples = None,
                 leaf_vars = [], depth = 1,
                 multilabel = False, n_labels=0, ml_tree_structure=0, xcnet=False):

       
        self.min_instances = min_instances
        self.min_features = min_features
        self.alpha = alpha
        self.depth = depth
        self.data = data
        self.node = TreeNode()
        self.multilabel = multilabel
        self.n_labels = n_labels
        self.ml_tree_structure = ml_tree_structure
        self.xcnet = xcnet

        self.leaf_vars = leaf_vars
        self.n = data.shape[1]

        if n_original_samples is None:
            self.n_original_samples = self.data.shape[0]
        else:
            self.n_original_samples = n_original_samples


        if clt is None:
            COC = [[] for i in range(data.shape[0])]
            for r in range(data.shape[0]):
                for f in range(data.shape[1]):
                    if data[r,f]>0:
                        COC[r].append(f)

            self.node.cltree = Cltree()

            
            self.node.cltree.fit(data, alpha=self.alpha, 
                                 multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure)

            self.orig_ll = self.node.cltree.score_samples_log_proba(self.data)
            sparsity = 0.0
            sparsity = len(self.data.nonzero()[0])
            sparsity /= (self.data.shape[1] * self.data.shape[0])
            logger.info("Dataset sparsity: %f", sparsity)
        else:
            self.node.cltree = clt
            self.orig_ll = ll

        self.scope = self.node.cltree.scope

       
        self.id = Csn._id_node_counter
        Csn._id_node_counter = Csn._id_node_counter + 1
        print("Block", self.id, "on", len(self.scope), "features and", self.data.shape[0], "instances, local ll:", self.orig_ll)

        if self.data.shape[0] > self.min_instances:
            if self.data.shape[1] >= self.min_features:
                self.or_cut()
            else:
                print( " > no cutting due to few features")
        else:
            print(" > no cutting due to few instances")

        if is_tree_node(self.node):
            if self.depth > Csn._depth:
                Csn._depth = self.depth
            Csn._mean_depth = Csn._mean_depth + self.depth
            Csn._leaf_nodes = Csn._leaf_nodes + 1
            Csn._cltrees = Csn._cltrees + 1
            Csn._clt_edges = Csn._clt_edges + self.node.cltree.num_edges

    def check_correctness(self,k):
        mean = 0.0
        for world in itertools.product([0,1], repeat=k):
            prob = np.exp(self._score_sample_log_proba(world))
            mean = mean + prob
        return mean 


    def show(self):
        """ WRITEME """
        print ("Learned Cut Set Network")
#        self._showl(0)
        print("OR nodes:", Csn._or_nodes)
        print("Leaves:", Csn._leaf_nodes)
        print("Cltrees:", Csn._cltrees)
        print("Edges outgoing OR nodes:", Csn._or_edges)
        print("Edges in CLtrees:", Csn._clt_edges)
        print("Total edges:", Csn._or_edges + Csn._clt_edges)
        print("Total nodes:", Csn._or_nodes + Csn._leaf_nodes + Csn._and_nodes)
        print("Depth:", Csn._depth)
        print("Mean Depth:", Csn._mean_depth / Csn._leaf_nodes)

    def _showl(self,level):
        """ WRITEME """
        if is_or_node(self.node):
            print(self.id,"OR", self.node.left_weight,self.node.left_child.id,self.node.right_child.id,"on",self.scope[self.node.or_feature])
            self.node.left_child._showl(level+1)
            self.node.right_child._showl(level+1)
        elif is_and_node(self.node):
            print(self.id, "AND", end="")
            for i in range(len(self.tree_forest)):
                if self.node.or_features[i] == None:
                    print("()", end="")
                else:
                    print("(",self.node.children_left[i].id,self.node.children_right[i].id,"on",self.node.cltree.scope[self.tree_forest[i][self.node.or_features[i]]],")", end="")
            print("")
            for i in range(len(self.tree_forest)):
                if self.node.or_features[i] is not None:
                    self.node.children_left[i]._showl(level+1)
                    self.node.children_right[i]._showl(level+1)
        elif is_sum_node(self.node):
            print(self.id,"SUM", self.node.weights)
            for c in self.node.children:
                c._showl(level+1)
        else:
            print(self.id, "LEAF", end=" ")
            if self.node.cltree.is_forest():
                print("Forest")
            else:
                print("Tree")
                print(self.node.cltree.tree)
                print(self.node.cltree.scope)


    def mpe(self, evidence = {}):
        """ WRITEME """
        return self.node.mpe(evidence)

    def marginal_inference(self, evidence = {}):
        """ WRITEME """
        return self.node.marginal_inference(evidence)

    def naiveMPE(self, evidence = {}):
        maxprob = -np.inf
        maxstate = []
        for w in (itertools.product([0, 1], repeat=self.n)):
            ver = True
            for var, state in evidence.items():
                if w[var] != state:
                    ver = False
                    break
            if ver:
                prob = self.score_sample_log_proba(w)
                print(prob)
                if prob > maxprob:
                    maxprob = prob
                    maxstate = w
        return (maxstate, maxprob)



    def score_sample_log_proba(self,x):
        return self.node.score_sample_log_proba(x)
        
    def score_samples_log_proba(self, X):
        Prob = X[:,0]*0.0
        for i in range(X.shape[0]):
            Prob[i] = self.score_sample_log_proba(X[i])

        m = np.sum(Prob) / X.shape[0]
        return m



    def score_samples_proba(self, X):
        Prob = X[:,0]*0.0
        for i in range(X.shape[0]):
            Prob[i] = np.exp(self.score_sample_log_proba(X[i]))
        return Prob
        

    def or_cut(self):
        print(" > trying to cut ... ")
        sys.stdout.flush()

        found = False

        bestlik = self.orig_ll
        best_clt_l = None
        best_clt_r = None
        best_feature_cut = None
        best_left_weight = 0.0
        best_right_weight = 0.0
        best_right_data = None
        best_left_data = None
        best_v_ll = 0.0
        best_gain = -np.inf
        best_left_sample_weight = None
        best_right_sample_weight = None
                            
        cutting_features = []
        for f in range(self.node.cltree.n_features):
            if self.scope[f] not in self.leaf_vars:
                cutting_features.append(f)
        
        selected = cutting_features

        if self.xcnet:
            selected = [random.choice(selected)]
            bestlik = -np.inf

        ll = 0.0
        CL_l = None 
        CL_r = None
        feature = None
        left_weight = 0.0
        right_weight = 0.0
        left_data = None
        right_data = None
        l_ll = 0.0 
        r_ll = 0.0
            
        for feature in selected:
            condition = self.data[:,feature]==0
            new_features = np.ones(self.data.shape[1], dtype=bool)
            new_features[feature] = False
            left_data = self.data[condition,:][:, new_features]
            right_data = self.data[~condition,:][:, new_features]
            left_weight = (left_data.shape[0] ) / (self.data.shape[0] )
            right_weight = (right_data.shape[0] ) / (self.data.shape[0] )        
            
            if left_data.shape[0] > 0 and right_data.shape[0] > 0:          

                left_scope = np.concatenate((self.node.cltree.scope[0:feature],self.node.cltree.scope[feature+1:]))
                right_scope = np.concatenate((self.node.cltree.scope[0:feature],self.node.cltree.scope[feature+1:]))
                CL_l = Cltree()
                CL_r = Cltree()

                CL_l.fit(left_data,scope=left_scope,alpha=self.alpha, 
                         multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure)
                CL_r.fit(right_data,scope=right_scope,alpha=self.alpha, 
                         multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure)

                l_ll = CL_l.score_samples_log_proba(left_data)
                r_ll = CL_r.score_samples_log_proba(right_data)

                ll = ((l_ll+np.log(left_weight))*left_data.shape[0] + (r_ll+np.log(right_weight))*right_data.shape[0])/self.data.shape[0]
            else:
                ll = -np.inf

            if ll>bestlik:

                bestlik = ll
                best_clt_l = CL_l
                best_clt_r = CL_r
                best_feature_cut = feature
                best_left_weight = left_weight
                best_right_weight = right_weight
                best_right_data = right_data
                best_left_data = left_data
                best_l_ll = l_ll
                best_r_ll = r_ll
                found = True

        gain = (bestlik - self.orig_ll)
        print ("   - gain cut:", gain, end = "")

        if (found==True):
            self.node = OrNode()
            self.node.or_feature_scope = self.scope[best_feature_cut]
            Csn._or_nodes = Csn._or_nodes + 1
            Csn._or_edges = Csn._or_edges + 2

            self.node.or_feature = best_feature_cut
            print("   - cutting on feature ", self.node.or_feature, "[#l:",best_left_data.shape[0],", #r:",best_right_data.shape[0],"], gain:", bestlik - self.orig_ll)

            instances = self.data.shape[0]

            self.node.left_weight = best_left_weight
            self.node.right_weight = best_right_weight

            # free memory before to recurse
            self.free_memory()

            self.node.left_child = Csn(data=best_left_data, 
                                       clt=best_clt_l, ll=best_l_ll, 
                                       min_instances=self.min_instances, 
                                       min_features=self.min_features, alpha=self.alpha, 
                                       leaf_vars = self.leaf_vars,
                                       n_original_samples = self.n_original_samples,
                                       depth=self.depth+1,
                                       multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure, xcnet=self.xcnet)
            self.node.right_child = Csn(data=best_right_data, 
                                        clt=best_clt_r, ll=best_r_ll, 
                                        min_instances=self.min_instances, 
                                        min_features=self.min_features, alpha=self.alpha,
                                        leaf_vars = self.leaf_vars,
                                        n_original_samples = self.n_original_samples,
                                        depth=self.depth+1,
                                        multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure, xcnet=self.xcnet)

        else:
            print(" no cutting")


    def free_memory(self):
        self.data = None
        self.validation = None
