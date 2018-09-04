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

class Node(object):
    """Base class for all nodes
    """
    _id_counter = 0

    def __init__(self):
        self.id = Node._id_counter
        Node._id_counter += 1

       
class OrNode(Node):
    """Class for or nodes
    """
    _node_type = "or"
    
    def __init__(self):
        Node.__init__(self)
        self.left_child = None
        self.right_child = None
        self.left_weight = 0.0
        self.right_weight = 0.0
        self.or_feature = None
        self.or_feature_scope = None


    def score_sample_log_proba(self, x):
        """ WRITEME """
        prob = 0.0
        x1 = np.concatenate((x[0:self.or_feature],x[self.or_feature+1:]))
        if x[self.or_feature] == 0:
            prob = prob + np.log(self.left_weight) + self.left_child.score_sample_log_proba(x1)
        else:
            prob = prob + np.log(self.right_weight) + self.right_child.score_sample_log_proba(x1)
        return prob

    def mpe(self, evidence={}):
        mpe_log_proba = 0.0
        state_evidence = evidence.get(self.or_feature_scope)
        if state_evidence is not None:
            if state_evidence == 0:
                (mpe_state, mpe_log_proba) = self.left_child.mpe(evidence)
                mpe_state[self.or_feature_scope] = 0
                mpe_log_proba += np.log(self.left_weight)
            else:
                (mpe_state, mpe_log_proba) = self.right_child.mpe(evidence)
                mpe_state[self.or_feature_scope] = 1
                mpe_log_proba += np.log(self.right_weight)
        else:
            (left_mpe_state, left_mpe_log_proba) = self.left_child.mpe(evidence)
            (right_mpe_state, right_mpe_log_proba) = self.right_child.mpe(evidence)
            if left_mpe_log_proba + np.log(self.left_weight) > right_mpe_log_proba + np.log(self.right_weight):
                mpe_state = left_mpe_state
                mpe_state[self.or_feature_scope] = 0
                mpe_log_proba = left_mpe_log_proba + np.log(self.left_weight)
            else:
                mpe_state = right_mpe_state
                mpe_state[self.or_feature_scope] = 1
                mpe_log_proba = right_mpe_log_proba + np.log(self.right_weight)
        return (mpe_state, mpe_log_proba)

    def marginal_inference(self, evidence={}):
        log_proba = 0.0
        state_evidence = evidence.get(self.or_feature_scope)
        if state_evidence is not None:
            if state_evidence == 0:
                log_proba = self.left_child.marginal_inference(evidence)
                log_proba += np.log(self.left_weight)
            else:
                log_proba = self.right_child.marginal_inference(evidence)
                log_proba += np.log(self.right_weight)
        else:
            left_log_proba = self.left_child.marginal_inference(evidence)
            right_log_proba = self.right_child.marginal_inference(evidence)
            log_proba = np.log(np.exp(left_log_proba)*self.left_weight + np.exp(right_log_proba)*self.right_weight)
        return log_proba


class TreeNode(Node):
    """Class for tree nodes
    """
    _node_type = "tree"

    def __init__(self):
        Node.__init__(self)
        self.cltree = None

    def score_sample_log_proba(self, x):
        """ WRITEME """
        return self.cltree.score_sample_log_proba(x)
            
    def mpe(self, evidence={}):
        return self.cltree.mpe(evidence)

    def marginal_inference(self, evidence={}):
        return self.cltree.marginal_inference(evidence)


###############################################################################

def is_or_node(node):
    """Returns True if the given node is a or node."""
    return getattr(node, "_node_type", None) == "or"

def is_tree_node(node):
    """Returns True if the given node is a tree node."""
    return getattr(node, "_node_type", None) == "tree"

