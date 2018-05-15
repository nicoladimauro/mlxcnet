import numpy as np
from logr import logr

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
            prob = prob + logr(self.left_weight) + self.left_child.score_sample_log_proba(x1)
        else:
            prob = prob + logr(self.right_weight) + self.right_child.score_sample_log_proba(x1)
        return prob

    def mpe(self, evidence={}):
        mpe_log_proba = 0.0
        state_evidence = evidence.get(self.or_feature_scope)
        if state_evidence is not None:
            if state_evidence == 0:
                (mpe_state, mpe_log_proba) = self.left_child.mpe(evidence)
                mpe_state[self.or_feature_scope] = 0
                mpe_log_proba += logr(self.left_weight)
            else:
                (mpe_state, mpe_log_proba) = self.right_child.mpe(evidence)
                mpe_state[self.or_feature_scope] = 1
                mpe_log_proba += logr(self.right_weight)
        else:
            (left_mpe_state, left_mpe_log_proba) = self.left_child.mpe(evidence)
            (right_mpe_state, right_mpe_log_proba) = self.right_child.mpe(evidence)
            if left_mpe_log_proba + logr(self.left_weight) > right_mpe_log_proba + logr(self.right_weight):
                mpe_state = left_mpe_state
                mpe_state[self.or_feature_scope] = 0
                mpe_log_proba = left_mpe_log_proba + logr(self.left_weight)
            else:
                mpe_state = right_mpe_state
                mpe_state[self.or_feature_scope] = 1
                mpe_log_proba = right_mpe_log_proba + logr(self.right_weight)
        return (mpe_state, mpe_log_proba)

    def marginal_inference(self, evidence={}):
        log_proba = 0.0
        state_evidence = evidence.get(self.or_feature_scope)
        if state_evidence is not None:
            if state_evidence == 0:
                log_proba = self.left_child.marginal_inference(evidence)
                log_proba += logr(self.left_weight)
            else:
                log_proba = self.right_child.marginal_inference(evidence)
                log_proba += logr(self.right_weight)
        else:
            left_log_proba = self.left_child.marginal_inference(evidence)
            right_log_proba = self.right_child.marginal_inference(evidence)
            log_proba = logr(np.exp(left_log_proba)*self.left_weight + np.exp(right_log_proba)*self.right_weight)
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

