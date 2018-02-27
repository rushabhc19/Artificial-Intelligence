"""This module contains class with methods to read data from features table"""
import os
import pandas as pd
import numpy as np

class Data(object):
    """Handles interactions with data"""

    def __init__(self):
        # The number of consecutive days to take into account for features
        self.history_window = 4

        # The number of days in the future, to predict for a flood
        self.predict_ahead = 7

    def func_i(self, feature_1, feature_2):
        """ This is the function that determines how related is one feature to
        another. It returns a z_score that will be used to see if the two
        features are related to one another.
        """

        if len(feature_1) != len(feature_2):
            raise ValueError("Compared features must have same length")

        # get pearson correlation coefficient of feature_1 with feature_2
        correlation = feature_1.corr(feature_2)

        # fisher transformation of the coefficient
        fisher_t = np.arctanh(correlation)

        # z_score of the fisher transform
        z_score = fisher_t * np.sqrt(len(feature_1) - 3)

        return abs(z_score)

    def get_feature_col(self, column_id):
        """ Retrieves requested column

        This method takes a tuple as only argument. The tuple should be of size 3.
        column_id = (minusd, variable, location_id)

        Returns a pandas.Series object with the requested feature column.

        The way to get the data is by reading the original column for a
        meteorological variable at specified location, and then shifting the index
        appropriately.
        """

        # unpack the input tuple
        minusd, variable, location_id = column_id

        # build the path name for the target file to read
        filepath = os.path.join('data', variable, str(location_id) + '.pickle')

        # read the pandas.Series column
        col = pd.read_pickle(filepath)
        # extract appropriate set of rows, according to minusd
        col = col.loc[self.history_window - minusd
                      :11310 - self.predict_ahead - minusd] # should be 11309

        # fix index
        col.index += minusd

        return col

    def get_label_col(self):
        """Returns a dataframe with the labels column"""

        # build the path name for the target file to read
        filepath = os.path.join('data', 'label', 'class.pickle')

        # read partial dataframe
        series = pd.read_pickle(filepath)
        series = series.loc[self.history_window + self.predict_ahead
                            :11310] # should be 11309
        series.index -= self.predict_ahead
        return series

    def get_sum15_col(self):
        """Returns a dataframe with the labels column"""

        # build the path name for the target file to read
        filepath = os.path.join('data', 'label', 'sum15.pickle')

        # read partial dataframe
        series = pd.read_pickle(filepath)
        series = series.loc[self.history_window + self.predict_ahead
                            :11310] # should be 11309
        series.index -= self.predict_ahead
        return series

    def write_col_pickles(self):
        """This method builds and writes the pickles for each of the columns
        that constitute the full table of features + class label"""

        # define indexes for features
        variables = ['pw', 't850', 'u300', 'u850', 'v300',
                     'v850', 'z1000', 'z300', 'z500']

        # make sure the necessary folders are available
        # if not, then create it and copy data
        for var in variables:
            directory = os.path.join('data', var)
            if not os.path.exists(directory):
                os.makedirs(directory)

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
                    filepath = os.path.join('data', var,
                                            str(location_id) + '.pickle')

                    print ("\033[FWriting pickle for location_id % 6d"
                           % location_id)
                    # save to pickle
                    mvardata[location_id].to_pickle(filepath)

        # now for class label column
        directory = os.path.join('data', 'label')
        if not os.path.exists(directory):
            os.makedirs(directory)

            #read Iowa data to extract the class labels
            iowa = self.__build_iowa_data()

            iowa['label'].to_pickle(os.path.join('data', 'label', 'class.pickle'))
            iowa['sum15'].to_pickle(os.path.join('data', 'label', 'sum15.pickle'))

        print "Process complete!!!"

        return

    def __build_iowa_data(self):
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

        # small fix to match TA's data... :(
        # his data includes this last row for labels
        sum15s.append(0.0)
        index = len(iowa) - 14
        for j in range(index, index + 14):
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
                class_labels.append(int(0))

        iowa = iowa.assign(label=pd.Series(class_labels, dtype='int8'))

        iowa.index += 1

        return iowa
