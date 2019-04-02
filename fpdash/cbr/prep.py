import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def openmlwrapper(data_id=31, random_state=1, n_samples = 2000, verbose=True, scale=True):
    """
    Wrapper for preprocessing OpenML datasets. Train/test split (75/25) and fill missing values with median of
    training set. 
    Optional: scale data through normalization (subtract mean, divide by standard deviation).
        
    Parameters
    ----------
    data_id : int
    openml dataset id
    random_state : int
        random state of the train test split
    n_samples : int
        number of samples from the data that will be returned
        
    Returns
    -------
    data_dict : dict
        Dictionary with data, including: X_train, X_test, y_train, y_test, X_train_decoded (original feature values), 
        X_test_decoded (original feature values)
    """
    dataset = openml.datasets.get_dataset(data_id)
    X, y, cat, att = dataset.get_data(target = dataset.default_target_attribute, 
                                             return_categorical_indicator=True,
                                             return_attribute_names=True)
    print('Start preprocessing...')
    # Sample at most n_samples samples
    if len(X) > n_samples:
        prng = np.random.RandomState(seed=1)
        rows = prng.randint(0, high=len(X), size=n_samples)
        X = X[rows, :]
        y = y[rows]
        if verbose:
            print("...Sampled %s samples from dataset %s." % (n_samples, data_id))
    else:
        if verbose:
            print("...Used all %s samples from dataset %s." % (len(X), data_id))
        
    # Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X, columns=att),
                                                        pd.DataFrame(y, columns=['class']),
                                                        random_state = random_state)
    # Fill missing values with median of X_train
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    if verbose:
        print('...Filled missing values.')
    
    # Create decoded version with original feature values for visualizations
    X_train_decoded = X_train.copy()
    X_test_decoded = X_test.copy()
    for f in att:
        labels = dataset.retrieve_class_labels(target_name=f)
        if labels != 'NUMERIC':
            labels_dict = {i : l for i,l in zip(range(len(labels)), labels)}
        else:
            labels_dict = {}
        X_test_decoded[f] = X_test_decoded[f].replace(labels_dict)
        X_train_decoded[f] = X_train_decoded[f].replace(labels_dict)
    if verbose:
        print('...Decoded to original feature values.')

    # Scale data
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=list(X_train))
        X_test = pd.DataFrame(scaler.transform(X_test), columns=list(X_test))
        if verbose:
            print('...Scaled data.')

        print('Preprocessing done.')
    return {'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'X_train_decoded' : X_train_decoded,
        'X_test_decoded' : X_test_decoded}

def plot_roc(y, y_score, label, max_fpr, xlim, mln = True):
    """
    Plot de ROC curve up to a particular maximum false positive rate.
    
    Parameters
    ----------
        y : array like [n_observations]
            true classes
        y_score : array like [n_observations]
            classification probabilities
        label : string
            dataset name
        max_fpr : numerical
            maximum false positive rate
        xlim : numerical
            limit of plot on x axis
        mln : Boolean
            display FPR per million

        Returns
        -------
        fpr : array
            fp rates
        tpr : array 
            tp rates
        thresholds : array
            prediction thresholds
    """
    ax = plt.axes()
    fpr, tpr, thresholds = roc_curve(y, y_score, drop_intermediate=False)
    plt.plot([0, 1], [0, 1], '--', linewidth=1, color='0.25')
    plt.plot(fpr, tpr, label = 'Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve %s data' % label)
    if xlim:
        if mln:
            plt.plot([max_fpr,max_fpr], [0, 1], 'r--', linewidth=1, label = 'FPR $\leq %.f*10^{-6}$'%(max_fpr*10**6))
            labels = plt.xticks()
            ax.set_xticklabels(['%.0f' %(i*10**6) for i in labels[0]])
            plt.xlabel('False Positive Rate (per 1 mln)')
        else:
            plt.plot([max_fpr, max_fpr], [0, 1], 'r--', linewidth=1, label = 'FPR $\leq %.2f$'%(max_fpr))
            plt.xlabel('False Positive Rate')
        plt.xlim(-.000001,xlim+.000001)
    plt.legend()
    plt.show()
    return fpr, tpr, thresholds