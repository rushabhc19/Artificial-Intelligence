#!/usr/bin/python
"""This module contains methods used to process raw data into the set of
features that will be considered by SAOLA algorithm"""
import os
import pandas as pd
import numpy as np

def build_iowa_data():
    """Builds a dataframe with precipitation data for Iowa. The resulting
    dataframe also includes a column with the sum of the next 15 days of
    precipitation for each day, and a column with class labels that are
    calculated according to sum15 column.
    """

    print "reading Iowa data..."

    # read the csv
    iowa = pd.read_csv(
        'raw_data/aggreg/iowa.csv',
        usecols=[4],
        header=None,
        names=['precip']
    )

    # this list will be used to build the new sum15 column for dataframe
    sum15s = []

    # iterate over dataframe
    for index in range(len(iowa) - 14):
        # initialize ith element of list
        sum15s.append(0.0)

        # iterate over the next 15 rows. We need to cap the upper bound
        # of the range so we dont overflow
        for j in range(index, index + 15):
            sum15s[index] += iowa['precip'][j]

    # extend the dataframe to include the new column
    iowa = iowa.assign(sum15=pd.Series(sum15s))

    # we now want to build class column.
    # finding threshold for 95 percentile
    threshold = iowa.quantile(q=0.95, axis=0)['sum15']

    # list used to build the column
    class_labels = []

    # iterate over dataframe
    for index in range(len(iowa)):
        if iowa['sum15'][index] >= threshold:
            # found a flood, this is a positive label
            class_labels.append(int(1))
        elif iowa['sum15'][index] < threshold:
            # negative label
            class_labels.append(int(0))
        else:
            # undefined label
            class_labels.append(int(-1))

    iowa = iowa.assign(label=pd.Series(class_labels, dtype='int8'))

    iowa.index += 1

    return iowa

def write_col_pickles():
    """This method builds and writes the pickles for each of the columns that
    constitute the full table of features + class label"""

    # Check if process already completed
    if os.path.exists(os.path.join('data', 'label', 'class.pickle')):
        # do not repeat the work. Leave it as it is
        return

    # define indexes for features
    variables = ['pw', 't850', 'u300', 'u850', 'v300',
                 'v850', 'z1000', 'z300', 'z500']

    # make sure the necessary folders are available
    for var in variables:
        directory = os.path.join('data', var)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # now for class label column
    directory = os.path.join('data', 'label')
    if not os.path.exists(directory):
        os.makedirs(directory)

    for var in variables:
        print "Reading variable %s...\n" % var
        mvardata = pd.read_csv(
            "raw_data/aggreg/%s.csv" % var,
            usecols=range(4, 5332),     # 4th col -> location_id=1
            header=None,                # manually setting col names
            names=range(1, 5329),       # names are simply location_id
        )

        # fix index
        mvardata.index += 1

        for location_id in range(1, 5328 + 1):
            # build the path name for the target file to write
            filepath = os.path.join('data', var, str(location_id) + '.pickle')

            print "\033[FWriting pickle for location_id % 6d" % location_id
            # save to pickle
            mvardata[location_id].to_pickle(filepath)

    #read Iowa data to extract the class labels
    iowa = build_iowa_data()

    # select only the rows for which we have features
    labels = iowa.loc[15:11309]

    # make the index match the corresponding day from which we try to predict
    labels.index -= 5

    labels['label'].to_pickle(os.path.join('data', 'label', 'class.pickle'))
    labels['sum15'].to_pickle(os.path.join('data', 'label', 'sum15.pickle'))
    print "Process complete!!!"

    return

def get_feature_col(column_id):
    """
    This method takes a tuple as only argument. The tuple should be of size 3.
    column_id = (minusd, variable, location_id)

    Returns a dataframe with the requested feature column.

    The way to get the data is by reading the original column for a
    meteorological variable at specified location, and then shifting the index
    appropriately.
    """

    # unpack the input tuple
    minusd, variable, location_id = column_id

    # build the path name for the target file to read
    filepath = os.path.join('data', variable, str(location_id) + '.pickle')

    # real partial dataframe
    col = pd.read_pickle(filepath).loc[10 - minusd:11304 - minusd]

    # fix index
    col.index += minusd

    return col

def get_label_col():
    """Returns a dataframe with the labels column"""

    # build the path name for the target file to read
    filepath = os.path.join('data', 'label', 'class.pickle')

    # real partial dataframe
    return pd.read_pickle(filepath)

def usage_example():
    """Example code"""

    # We can load any column of the Main Table (features + label) like this
    col_id = (9, 'pw', 1)
    feat_col = get_feature_col(col_id)

    # Load the class label column
    label_col = get_label_col()
    #label_col = pd.read_pickle('data/label/sum15.pickle')      # ignore this

    # One very important process that we need, is to calculate correlation
    # between any two columns in the features + label table
    correlation = feat_col.corr(label_col)      # simple enough, isn't it?
                                                # and it's fast too
    print "(%d,% 6s,% 5d) corr w/ label: %s" % (9, 'pw', 1, correlation)

    variables = ['pw', 't850', 'u300', 'u850', 'v300',
                 'v850', 'z1000', 'z300', 'z500']

    print ("\nNow with Fisher's transformation and z-scores, "
           "for a set of random samples\n")

    # select history data
    for dminus in np.random.randint(0, 10, 3):

        # select meteorological variables
        for var in np.random.choice(variables, 3):

            # select locations
            for location_id in np.random.randint(1, 5328 + 1, 3):
                print ("\033[FCalculating correlation for (%d,% 6s,% 5d)"
                       % (dminus, var, location_id))

                # fetch column
                col_id = (dminus, var, location_id)
                feat_col = get_feature_col(col_id)

                # calculate correlation between selected feature and class label
                correlation = feat_col.corr(label_col)

                # Fisher's transformation for this correlation
                fisher_t = np.arctanh(correlation)

                # z-score is just dividing by standard deviation. In this case
                # standard deviation = 1/sqrt(N - 3)
                z_score = fisher_t * np.sqrt(11295 - 3)

                #report
                print ("\033[F(%d,% 6s,% 5d) corr w/ label: % 3.4f, "
                       "fisher-t: % 3.4f, z-score: % 8.4f\n"
                       % (dminus, var, location_id,
                          correlation, fisher_t, z_score))


if __name__ == "__main__":
    # Use this to build pickles
    write_col_pickles()
    usage_example()
