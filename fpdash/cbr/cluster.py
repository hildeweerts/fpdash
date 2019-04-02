# cluster imports
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

# cluster description imports
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.stats import uniform
from sklearn.metrics import roc_auc_score

# prototyping imports
import pandas as pd
import numpy as np

def wrapper(alg, n_clusters, instances, clu_params={}):
    """
    sklearn cluster wrapper for easy testing.
        
        Parameters
        ----------
        alg : str
            algorithm
        n_clusters : int
            number of clusters/components
        instances : array like size [n_instances, n_features]
            instances that need to be clustered
    """
    if alg=='gmm':
        clu = GaussianMixture(n_components=n_clusters, random_state=1, **clu_params)
    if alg=='spec':
        clu = SpectralClustering(n_clusters=n_clusters, random_state=1, **clu_params)
    if alg=='aggl':
        clu = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', **clu_params)
    if alg=='kmeans':
        clu = KMeans(n_clusters=n_clusters, **clu_params)
    labels = clu.fit_predict(instances)
    return clu, labels


class Prototype:
    """
    Prototypical explanation.
    """
    def __init__(self, label, feature_values):
        """
        Initialize prototype.
            Parameters
            ----------
            label : object
                label of the prototype
            feature_values : object of type dict, list or np.ndarray
                feature values of the prototype
        """
        self.label = label
        if (type(feature_values) == list) or (type(feature_values) == np.ndarray):
            self.feature_values = {i : v for i, v in zip(range(len(feature_values)), feature_values)}
        elif type(feature_values) == dict:
            self.feature_values = feature_values
        else:
            raise ValueError('Invalid feature values type.')
        return
    
class PrototypeSet:
    """
    Set of prototypical explanations.
    """
    def __init__(self, prototypes = None):
        """
        Initialize Prototype set.
        
        Parameters
        ----------
            prototypes : dictionary of Prototype instances
        """
        if prototypes is None:
            self.prototypes = {}
        else:
            self.prototypes = {p.label : p for p in prototypes}
        return
    
    def fit(self, X, labels, metric = 'mean'):
        """
        Fit prototypes based on a clustering.
        
        Parameters
        ----------
            X : array like [n_instances, n_features]
                actual values of the instances
            labels : array of int of size n_instances
                nearest prototypes
            metric : str
                metric used to compute prototypes
                    - 'mean' : average feature values of all instances in cluster
        """
        df = pd.DataFrame(X)
        df['label'] = labels
        for label, group in df.groupby('label'):
            if metric == 'mean':
                group = group.drop('label', axis=1)
                values = group.mean().to_dict()
            self.prototypes[label] = Prototype(label, values)
        return

def prototype_rmse(X, labels, ps):
    """
    Compute the RSME of a prototype set. This is the root mean squared error in case 
    SHAP explanations are predicted based on the SHAP explanation of the prototypes.
    
    Parameters
    ----------
        X : array like [n_instances, n_features]
        labels : array of int of size n_instances
            nearest prototypes
        ps : PrototypeSet
            set of prototypes which we like to evaluate
        feature_rsme : boolean
            retrieve RMSE on a per-feature basis
    """
    df = pd.DataFrame(X)
    df['label'] = labels
    se_total = 0
    for label, group in df.groupby('label'):
        group = group.drop('label', axis=1)
        p = ps.prototypes[label]
        # compute squared error for this prototype
        se = np.sum(np.sum(np.square(group - p.feature_values)))
        # add to total rmse, weighted by group size
        se_total += se
    rmse = np.sqrt(se_total/len(X))
    return rmse

def prototype_r2(X, labels, ps):
    """
    Compute the R2 of a prototype set. This is the amount of variance in the prediction probability
    that is explained by the prototype set.
    
    Parameters
    ----------
        X : array like [n_instances, n_features]
        labels : array of int of size n_instances
            nearest prototypes
        ps : PrototypeSet
            set of prototypes which we like to evaluate
    """
    df = pd.DataFrame(X)
    # Compute SS_tot
    y_i = df.sum(axis=1)
    y_mean = y_i.mean()
    SS_tot = np.sum(np.square(y_i - y_mean))
    
    # Compute SS_res
    df['label'] = labels
    SS_res = 0
    for label, group in df.groupby('label'):
        f_i = sum(ps.prototypes[label].feature_values.values())
        group = group.drop('label', axis=1)
        y_i_group = group.sum(axis=1)
        SS_res += np.sum(np.square(y_i_group - f_i))
        
    # Compute R2
    R2 = 1 - (SS_res/SS_tot)
    return R2

def fit_description_tree(clu, cluster, X, feature_names = None, random_state=1, 
             param_grid=None):
    """
    For a cluster within a clustering, fit a decision tree with target variable cluster membership.

        Parameters
        ----------
        clu : cluster object
            sklearn clustering object including labels_
        cluster : int
            cluster id in clu.labels_
        X : array like
            training instances
        feature_names : array like
            array containing feature names (str); if not passed X must be a pandas dataframe
        random_state : int
            random seed passed to decision tree and grid search
        param_distr : dict
            dictionary with parameter distributions for hyperparameter tuning
            
        Returns
        -------
        dt : DecisionTreeClassifier
            Description tree
        score : float
            AUC of decision tree classifier
        labels_dummy : list
            cluster membership (1 if in cluster, else 0)
    """
    
    # feature names
    if feature_names is None:
        feature_names = list(X)
    # set param_grid
    if param_grid is None:
        param_grid = {'max_depth' : list(range(1, 2*len(feature_names))), 
                      'class_weight' : ['balanced']}
    
    # retrieve labels
    labels_dummy = [1 if l == cluster else 0 for l in clu.labels_]
    
    # perform grid search
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=i)
    
    rscv = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_state),
                        param_grid=param_grid, scoring = 'roc_auc', 
                        iid=False, n_jobs=-1, cv=StratifiedKFold(3, shuffle=True, random_state=random_state), refit=True)
    rscv.fit(X, labels_dummy)
    dt = rscv.best_estimator_
    
    # evaluate estimator
    y_score = [i[1] for i in dt.predict_proba(X)]
    score = roc_auc_score(y_true=labels_dummy, y_score=y_score)
    return dt, score, labels_dummy

def get_description(tree, feature_names, labels_dummy = None, probability_threshold = 0.5):
    """
     Produce description for decision tree.
     
         Parameters
         -----------
         tree : scikit-learn DecisionTreeClassifier
             decision tree you want to describe
         feature_names : list 
             feature names
         labels_dummy : list
             dummy labels necessary to compute class weights when class_weight is 'balanced'
             1 : part of cluster, 0 : not part of cluster
         probability_threshold : float
             if proportion of instances belonging to the positive class > probability_threshold, the description
             is added to the set of descriptions
             
         Returns
         -------
         descriptions : dict
             dictionary with the descriptions
    """
    class_weight = tree.class_weight
    if class_weight == 'balanced':
        class_weight_vec = compute_class_weight(class_weight, [0,1], labels_dummy)
        class_weight = {0 : class_weight_vec[0], 1 : class_weight_vec[1]}
    descriptions = []
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature]
    
    def recurse(node, depth, descr):
        if tree_.feature[node] != _tree.TREE_UNDEFINED: # if internal node
            name = feature_name[node]
            threshold = tree_.threshold[node]
            descr_left = descr.copy()
            descr_left[node] = {'name' : name, 'threshold' : threshold, 'sign' : '<='}
            descr_right = descr.copy()
            descr_right[node] = {'name' : name, 'threshold' : threshold, 'sign' : '>'}
            recurse(tree_.children_left[node], depth + 1, descr_left)
            recurse(tree_.children_right[node], depth + 1, descr_right)
        else: #if leaf node
            value = tree_.value[node][0]
            value_0 = value[0]/class_weight[0] # number of instances not in cluster
            value_1 = value[1]/class_weight[1] # number of instances in cluster
            if value_1/(value_0 + value_1) > probability_threshold: # if leaf node belongs to target cluster:
                descriptions.append((descr, value_0, value_1))        
    recurse(0,1,{})
    return descriptions




"""
OLDER STUFF
"""

def prototype_r2_old(X, labels, ps):
    """
    Compute the accuracy of a prototype set.
    
    Parameters
    ----------
        X : array like [n_instances, n_features]
        labels : array of int of size n_instances
            nearest prototypes
        ps : PrototypeSet
            set of prototypes which we like to evaluate
    """
    df = pd.DataFrame(X)
    SS_tot = np.sum(np.sum(np.square(df - df.mean())))
    df['label'] = labels
    SS_res = 0
    for label, group in df.groupby('label'):
        group = group.drop('label', axis=1)
        p = ps.prototypes[label]
        SS_res += np.sum(np.sum(np.square(group - p.feature_values)))
    R2 = 1 - (SS_res / SS_tot)
    return R2

def get_description_old(tree, feature_names, labels_dummy = None):
    """
     Produce description for decision tree.
     
         Parameters
         -----------
         tree : scikit-learn DecisionTreeClassifier
             decision tree you want to describe
         feature_names : list 
             feature names
         labels_dummy : list
             dummy labels necessary to compute class weights when class_weight is 'balanced'
             1 : part of cluster, 0 : not part of cluster
    """
    class_weight = tree.class_weight
    if class_weight == 'balanced':
        class_weight_vec = compute_class_weight(class_weight, [0,1], labels_dummy)
        class_weight = {0 : class_weight_vec[0], 1 : class_weight_vec[1]}
    descriptions = []
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature]
    
    def recurse(node, depth, descr):
        if tree_.feature[node] != _tree.TREE_UNDEFINED: # if internal node
            name = feature_name[node]
            threshold = tree_.threshold[node]
            descr_left = descr.copy()
            descr_left[node] = {'name' : name, 'threshold' : threshold, 'sign' : '<='}
            descr_right = descr.copy()
            descr_right[node] = {'name' : name, 'threshold' : threshold, 'sign' : '>'}
            recurse(tree_.children_left[node], depth + 1, descr_left)
            recurse(tree_.children_right[node], depth + 1, descr_right)
        else: #if leaf node
            value = tree_.value[node][0]
            value_0 = value[0]/class_weight[0] # number of instances not in cluster
            value_1 = value[1]/class_weight[1] # number of instances in cluster
            if value_1 > value_0: # if leaf node belongs to target cluster:
                descriptions.append((descr, value_0, value_1))        
    recurse(0,1,{})
    return descriptions

def describe_old(clu, dt_params, X, feature_names = None, cluster = None):
    """
    For each cluster, fit a decision tree to describe the cluster.

        Parameters
        ----------
        clu : cluster object
            sklearn clustering object
        dt_params : dict
            decision tree parameters
        cluster : int
            cluster id of cluster you want to describe if you don't want to describe all clusters
        X : array like
            training instances (either in SHAP space or in feature value space)
    """
    # default class_weight
    if 'class_weight' not in dt_params:
        dt_params['class_weight'] = {0: 1, 1: 1} 
    descriptions = {}
    
    if feature_names is None:
        feature_names = list(X)

    # determine range
    if cluster is None:
        c_range = range(clu.n_clusters)
    else:
        c_range = [cluster]

    # compute descriptions
    for i in c_range:
        labels_dummy = [1 if l == i else 0 for l in clu.labels_]
        dt = DecisionTreeClassifier(**dt_params)
        dt.fit(X, labels_dummy)
        descriptions[i] = get_description(dt, feature_names, labels_dummy)
    return descriptions