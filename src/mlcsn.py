import numpy as np
import csn as CSN
import itertools

###############################################################################
class mlcsn:
    
    def __init__(self, data, min_instances=5, min_features=3, alpha=1.0, leaf_vars = [],
                 multilabel = True, n_labels=0, ml_tree_structure=0, xcnet=False):

        self.data = data
        self.min_instances = min_instances
        self.min_features = min_features
        self.leaf_vars = leaf_vars
        self.alpha = alpha

        self.or_nodes = 0.0
        self.leaf_nodes = 0.0
        self.or_edges = 0.0
        self.clt_edges = 0.0
        self.cltrees = 0.0
        self.depth = 0.0
        self.mdepth = 0.0

        self.csn = None

        self.n_labels = n_labels
        self.multilabel = multilabel
        self.ml_tree_structure = ml_tree_structure
        self.xcnet = xcnet
   
    def fit(self):
        
        CSN.Csn.init_stats()

        self.csn = CSN.Csn(data=self.data,
                           n_original_samples = self.data.shape[0],
                           min_instances=self.min_instances, min_features=self.min_features, alpha=self.alpha, 
                           leaf_vars = self.leaf_vars,
                           depth = 1, 
                           multilabel = self.multilabel, n_labels=self.n_labels, ml_tree_structure=self.ml_tree_structure, xcnet=self.xcnet)

        self.ll = self.csn.score_samples_log_proba(self.data)
        self.or_nodes = CSN.Csn._or_nodes 
        self.leaf_nodes = CSN.Csn._leaf_nodes
        self.or_edges = CSN.Csn._or_edges
        self.clt_edges = CSN.Csn._clt_edges
        self.cltrees = CSN.Csn._cltrees
        self.depth = CSN.Csn._depth
        self.mdepth = CSN.Csn._mean_depth / CSN.Csn._leaf_nodes

    def score_samples(self, data, out_filename):
        with open(out_filename, 'w') as out_log:
            mean = 0.0
            for x in data:
                prob = self.csn.score_sample_log_proba(x)
                mean = mean + prob
                out_log.write('%.10f\n'%prob)
        out_log.close()
        return mean / data.shape[0]

    def mpe(self, evidence = {}):
        return self.csn.mpe(evidence)

    def naiveMPE(self, evidence = {}):
        return self.csn.naiveMPE(evidence)

    def compute_predictions(self, X, n_labels):
        predictions = np.zeros((X.shape[0],n_labels),dtype=np.int)
        n_attributes = X.shape[1]
        k = 0
        for x in X:
            evidence = {}
            for i in range(n_attributes):
                evidence[i]=x[i]

            (state, prob) = self.mpe(evidence = evidence)

            sum = 0
            for i in range(n_attributes, n_attributes + n_labels):
                sum += state[i]
            if sum == 0:
                # avoiding empty predictions
                max_state = None
                max_prob = -np.inf
                for i in range(n_attributes, n_attributes + n_labels):
                    evidence[i] = 1
                    if i > (n_attributes):
                        del evidence[i-1]
                    (state1, prob1) = self.mpe(evidence = evidence)
                    if (prob1 > max_prob):
                        max_prob = prob1
                        max_state = state1
                state = max_state
                prob = max_prob
            y = 0
            for i in range(n_attributes, n_attributes + n_labels):
                predictions[k,y]=state[i]
                y += 1
            k += 1
        return predictions

    def compute_predictions1(self, X, n_labels):
        predictions = np.zeros((X.shape[0],n_labels),dtype=np.int)
        n_attributes = X.shape[1]
        x1 = np.zeros(n_attributes + n_labels, dtype=np.int)

        probs = np.zeros((n_labels,2))
        
        k = 0
        for x in X:
            for j in range(n_attributes):
                x1[j] = x[j]
            for l in range(n_labels):
                prob0 = 0.0
                prob1 = 0.0
                # marginalize over all the other labels
                for state in itertools.product([0, 1], repeat=n_labels-1):
                    for j in range(n_labels):
                        if j != l:
                            if j<l:
                                x1[n_attributes+j] = state[j]
                            else:
                                x1[n_attributes+j] = state[j-1]
                    x1[n_attributes+l] = 0
                    prob0 += np.exp(self.csn.score_sample_log_proba(x1))
                    x1[n_attributes+l] = 1
                    prob1 += np.exp(self.csn.score_sample_log_proba(x1))
                probs[l,0] = prob0
                probs[l,1] = prob1
                predictions[k,l] = np.argmax(probs[l])


            if np.sum(predictions[k]) == 0:
                predictions[k, np.argmax(probs[:,1])] = 1
            k += 1

        return predictions

    def marginal_inference(self, X, n_labels):
        predictions = np.zeros((X.shape[0],n_labels),dtype=np.int)
        n_attributes = X.shape[1]
        x1 = np.zeros(n_attributes + n_labels, dtype=np.int)

        probs = np.zeros((n_labels,2))
        

        k = 0
        for x in X:
            for j in range(n_attributes):
                x1[j] = x[j]
            for l in range(n_labels):
                D = {}
                for j in range(n_attributes):
                    D[j] = x[j]
                D[l+n_attributes]=0
                prob0 = self.csn.marginal_inference(D)
                D[l+n_attributes]=1
                prob1 = self.csn.marginal_inference(D)
                probs[l,0] = prob0
                probs[l,1] = prob1
                predictions[k,l] = np.argmax(probs[l])

            if np.sum(predictions[k]) == 0:
                predictions[k, np.argmax(probs[:,1])] = 1
            k += 1

        return predictions

    def compute_probs(self, data):


        probs_XY = np.zeros(data['X'].shape[0])
        probs_X = np.zeros(data['X'].shape[0])
        probs_Y = np.zeros(data['X'].shape[0])
        probs_Y_given_X = np.zeros(data['X'].shape[0])

        XY = np.concatenate((data['X'],data['Y']), axis = 1)
        X = data['X']
        Y = data['Y']
        

        for i in range(X.shape[0]):
            log_prob_xy = self.csn.score_sample_log_proba(XY[i])

            D = {}
            for j in range(X.shape[1]):
                D[j] = X[i,j]

            log_prob_x = self.csn.marginal_inference(D)
            D = {}
            for j in range(Y.shape[1]):
                D[j+X.shape[1]] = Y[i,j]

            log_prob_y = self.csn.marginal_inference(D)

            probs_XY[i] = log_prob_xy
            probs_X[i] = log_prob_x
            probs_Y[i] = log_prob_y
            probs_Y_given_X[i] = probs_XY[i] - probs_X[i]

        return(probs_XY, probs_X, probs_Y, probs_Y_given_X)
            
