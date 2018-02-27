#!/usr/bin/python
"""Module docstring"""
import os
import numpy as np
import pandas as pd
from pickle_data_2 import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import AllKNN
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def main():
    """The main method"""
    #####################
    ### CONFIGURATION ###
    #####################

    # Agent() attempts to process samples in order to enhance another classifier
    classifier = Agent()

    # The classifier used by the agent after processing samples
    classifier.clf = LogisticRegression()
    classifier.clf.name = "LogisticRegression"  # attribute used for string formating

    # Take into account calendar data, so we can filter samples on this criteria.
    classifier.reject_by_calendar = False
    # Some classifiers can take weights for each sample, so we can give more
    # importance the the minority class
    classifier.use_weights = False
    # this agent can use a combination of undersampling the majority class and then
    # oversampling the minority class
    classifier.use_resampling = True

    # wheter to plot results for 1803 test samples (True), or for each CV test set
    use_test_samples = False

    #options = [False, True]
    #for rbc in options:
    #    classifier.reject_by_calendar = rbc
    #    for uwg in options:
    #        classifier.use_weights = uwg
    #        for urs in options:
    #            classifier.use_resampling = urs
    #            results(classifier, use_test_samples)
    results(classifier, use_test_samples)

def results(classifier, use_test_samples):
    #######################
    ### PREPARE SAMPLES ###
    #######################

    # read and split data
    training_data, test_data = prepare_data()

    # the third value here is data about corresponding month for each sample
    tra_x, tra_y, tra_c = training_data     # first 9497 samples
    tst_x, tst_y, tst_c = test_data         #  next 1803 samples

    ########################
    ### CROSS VALIDATION ###
    ########################
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    # helper object to split data for 10-fold cross validation
    cross_validator = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    maxauc = 0.0
    minauc = 1.0
    for train, test in cross_validator.split(tra_x, tra_y):
        if use_test_samples:
            # use samples from test set (1803 samples)
            topredict_x = tst_x
            topredict_y = tst_y
            topredict_c = tst_c
        else:
            # use test set for this cross validation iteration
            topredict_x = tra_x[test]
            topredict_y = tra_y[test]
            topredict_c = tra_c[test]

        classifier.fit(tra_c[train], tra_x[train], tra_y[train])
        # scores are a level of confidence for determining label
        scores = classifier.predict_proba(topredict_c, topredict_x)[:, 1]
        # predictions calculated from scores. 0.5 is the threshold
        pred = np.array([1 if p >= 0.5 else 0 for p in scores])
        conf = confusion_matrix(topredict_y, pred)
        confstr = ("Fold %d --> [TN:%5d, FP:%5d, FN:%5d, TP:%5d]" 
                   % (i, conf[0][0], conf[0][1], conf[1][0], conf[1][1]))
        print ("%s    Accuracy score:%7.2f %%"
               % (confstr, accuracy_score(topredict_y, pred) * 100))
        fpr, tpr, thresholds = roc_curve(topredict_y, scores)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        if roc_auc > maxauc:
            maxconfstr = confstr
            maxauc = roc_auc
        if roc_auc < minauc:
            minconfstr = confstr
            minauc = roc_auc
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plot_title = 'ROC for %s with %s' % (classifier.name, classifier.clf.name)
    plot_title += ('\nUseCalendar: %s, UseWeigths: %s, UseResampling: %s'
                   % (classifier.reject_by_calendar,
                      classifier.use_weights,
                      classifier.use_resampling))
    plot_title += '\n%s    %s' % (minconfstr, maxconfstr)

    plt.title(plot_title)
    plt.legend(loc="lower right")

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    plt.show()

def normalize(data):
    """Returns data with columns normalized
    input: numpy array
    output: numpy array
    """
    # normalize data and return
    # https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
    return (data - data.min(axis=0)) / data.ptp(axis=0)

def load_ta_data():
    """Reads datafile and returns data as numpy array"""
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.astype.html
    data = np.load("data/data_selected_1980_2010.npy").astype(float)
    return normalize(data)

def load_ta_target():
    """Reads target labels and returns two columns: sum15 and label"""
    filepath = os.path.join("data", "target_1980_2010.npy")
    target = np.load(filepath)
    return target[:, 1]

def split_samples(data):
    """Splits data into training samples and test samples
    input: numpy array

    returns tuple (training_samples, test_samples)
    both are numpy arrays
    """

    training_samples = data[0:9497]
    test_samples = data[9497:11300]

    return training_samples, test_samples

def prepare_data():
    """Prepare data for classifier to use"""
    data, label = load_ta_data(), load_ta_target()
    calendar = Data().get_months_cols().values
    tra_x, tst_x = split_samples(data)
    tra_y, tst_y = split_samples(label)
    tra_c, tst_c = split_samples(calendar)
    training_data = [tra_x, tra_y, tra_c]
    test_data = [tst_x, tst_y, tst_c]
    print "Loaded %d features" % len(tra_x[0])
    return [training_data, test_data]

class Agent(object):
    def __init__(self):
        self.name = "Special Agent"
        self.gnb = GaussianNB()
        #self.clf = SVC(probability=True)
        #self.clf.name = "SVM"
        self.clf = GaussianNB()
        self.clf.name = "GaussianNB"
        self.use_weights = False
        self.use_resampling = False
        self.reject_by_calendar = False
        self.samplesize = []

    def fit(self, c_data, x_data, y_data):
        # this is to track evolution of the size of the training samples
        self.samplesize = []
        self.samplesize.append(len(x_data))

        if self.reject_by_calendar:
            mask = self.mask_cal(c_data, y_data)
            # filter rows rejected by this calendar criteria
            # not filtering them might improve second classifier training
            #x_data = normalize(x_data[mask])
            #y_data = y_data[mask]
            self.samplesize.append(len(x_data))

        if self.use_resampling:
            # undersample
            resampler = AllKNN()
            x_data, y_data = resampler.fit_sample(x_data, y_data)
            self.samplesize.append(len(x_data))

            # oversample
            resampler = SMOTEENN()
            x_data, y_data = resampler.fit_sample(x_data, y_data)
            self.samplesize.append(len(x_data))

        # train clf only with filtered and resampled data
        if self.use_weights:
            try:
                self.clf.fit(x_data, y_data, self.get_weights(y_data))
            except TypeError:
                print "The classifier selected does not admit weights for training samples"
                print "Switching to no weights"
                self.use_weights = False
                self.clf.fit(x_data, y_data)
        else:
            self.clf.fit(x_data, y_data)

    def predict(self, c_data, x_data):
        """Returns predictions, given features and calendar data"""
        if self.reject_by_calendar:
            cal_pred = np.array(self.gnb.predict(c_data))
        else:
            cal_pred = np.full((len(x_data),), 1, dtype=np.int)
        clf_pred = np.array(self.clf.predict(x_data))
        # return logical and: cal_pred AND clf_pred
        # this means that both criterias have to agree in order to classify
        # a sample as positive
        return clf_pred * cal_pred

    def predict_proba(self, c_data, x_data):
        """Returns confidence values for predictions"""
        if self.reject_by_calendar:
            cal_proba = np.array(self.gnb.predict_proba(c_data))
        else:
            cal_proba = np.full((len(x_data), 2), 1, dtype=np.int)
        clf_proba = np.array(self.clf.predict_proba(x_data))
        return clf_proba * cal_proba

    def mask_cal(self, calendar_data, target_data):
        """The resulting array is a list of indices that exclude some of the samples"""
        self.gnb.fit(calendar_data, target_data)
        gnb_pred = self.gnb.predict(calendar_data)
        return np.array([i for i, v in enumerate(gnb_pred) if v > 0])

    def get_weights(self, y_data):
        if not self.use_weights:
            return np.full((len(y_data),), 1, dtype=np.int)
        class_1_weight = 1.0 - float(sum(y_data))/len(y_data)
        class_0_weight = (1.0 - class_1_weight)/1.5
        w_data = [class_1_weight if y == 1 else class_0_weight for y in y_data]
        return np.array(w_data)

if __name__ == "__main__":
    main()