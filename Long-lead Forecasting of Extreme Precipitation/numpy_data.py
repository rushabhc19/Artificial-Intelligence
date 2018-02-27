"""This module contains methods to read data from features table"""
import os
import pandas as pd
import numpy as np

class Data(object):
    """Handles interactions with data"""

    def __init__(self):
        # Will hold the currently active dataset data
        self.__data = None

        # Tracks the id of the currently loaded dataset
        self.__loaded = None

        # The number of consecutive days to take into account for features
        self.history_window = 4

        # The number of days in the future, to predict for a flood
        # I think that TA changed this from 5 to 7 days in the future
        self.predict_ahead = 7

    def func_i(self, feature_1, feature_2, return_fisher_t=False):
        """ This is the function that determines how related is one feature to
        another. It returns a z_score that will be used see if the two features
        are related to one anouther.
        """

        if len(feature_1) != len(feature_2):
            raise Exception("Compared features must have same length")

        # get pearson correlation coefficient of feature_1 with feature_2
        correlation = feature_1.corr(feature_2)

        # fisher transform of the coefficient
        fisher_t = np.arctanh(correlation)

        # z_score of the fisher transform
        z_score = fisher_t * np.sqrt(len(feature_1) - 3)

        if return_fisher_t:
            return abs(fisher_t)
        else:
            return abs(z_score)

    def __map(self, column_id):
        """This returns the id of the datafile in which the requested column
        is located"""

        # unpack the input tuple
        minusd, variable, location_id = column_id

        if variable == 'z300' and 1 <= location_id <= 4795:
            # data0[0,    0] = (9, ' z300',    1)
            # data0[0, 4794] = (9, ' z300', 4795)
            return 0, 0 + location_id - 1, minusd
        if variable == 'z300' and 4796 <= location_id <= 5328:
            # data1[0,    0] = (9, ' z300', 4796)
            # data1[0,  532] = (9, ' z300', 5328)
            return 1, 0 + location_id - 4796, minusd
        if variable == 'v850' and 1 <= location_id <= 4262:
            # data1[0,  533] = (9, ' v850',    1)
            # data1[0, 4794] = (9, ' v850', 4262)
            return 1, 533 + location_id - 1, minusd
        if variable == 'v850' and 4263 <= location_id <= 5328:
            # data2[0,    0] = (9, ' v850', 4263)
            # data2[0, 1065] = (9, ' v850', 5328)
            return 2, 0 + location_id - 4263, minusd
        if variable == 'u300' and 1 <= location_id <= 3729:
            # data2[0, 1066] = (9, ' u300',    1)
            # data2[0, 4794] = (9, ' u300', 3729)
            return 2, 1066 + location_id - 1, minusd
        if variable == 'u300' and 3730 <= location_id <= 5328:
            # data3[0,    0] = (9, ' u300', 3730)
            # data3[0, 1598] = (9, ' u300', 5328)
            return 3, 0 + location_id - 3730, minusd
        if variable == 'z1000' and 1 <= location_id <= 3196:
            # data3[0, 1599] = (9, 'z1000',    1)
            # data3[0, 4794] = (9, 'z1000', 3196)
            return 3, 1599 + location_id - 1, minusd
        if variable == 'z1000' and 3197 <= location_id <= 5328:
            # data4[0,    0] = (9, 'z1000', 3197)
            # data4[0, 2131] = (9, 'z1000', 5328)
            return 4, 0 + location_id - 3197, minusd
        if variable == 'u850' and 1 <= location_id <= 2663:
            # data4[0, 2132] = (9, ' u850',    1)
            # data4[0, 4794] = (9, ' u850', 2663)
            return 4, 2132 + location_id - 1, minusd
        if variable == 'u850' and 2664 <= location_id <= 5328:
            # data5[0,    0] = (9, ' u850', 2664)
            # data5[0, 2664] = (9, ' u850', 5328)
            return 5, 0 + location_id - 2664, minusd
        if variable == 'z500' and 1 <= location_id <= 2130:
            # data5[0, 2665] = (9, ' z500',    1)
            # data5[0, 4794] = (9, ' z500', 2130)
            return 5, 2665 + location_id - 1, minusd
        if variable == 'z500' and 2131 <= location_id <= 5328:
            # data6[0,    0] = (9, ' z500', 2131)
            # data6[0, 3197] = (9, ' z500', 5328)
            return 6, 0 + location_id - 2131, minusd
        if variable == 'pw' and 1 <= location_id <= 1597:
            # data6[0, 3198] = (9, '   pw',    1)
            # data6[0, 4794] = (9, '   pw', 1597)
            return 6, 3198 + location_id - 1, minusd
        if variable == 'pw' and 1598 <= location_id <= 5328:
            # data7[0,    0] = (9, '   pw', 1598)
            # data7[0, 3730] = (9, '   pw', 5328)
            return 7, 0 + location_id - 1598, minusd
        if variable == 'v300' and 1 <= location_id <= 1064:
            # data7[0, 3731] = (9, ' v300',    1)
            # data7[0, 4794] = (9, ' v300', 1064)
            return 7, 3731 + location_id - 1, minusd
        if variable == 'v300' and 1065 <= location_id <= 5328:
            # data8[0,    0] = (9, ' v300', 1065)
            # data8[0, 4263] = (9, ' v300', 5328)
            return 8, 0 + location_id - 1065, minusd
        if variable == 't850' and 1 <= location_id <= 531:
            # data8[0, 4264] = (9, ' t850',    1)
            # data8[0, 4794] = (9, ' t850',  531)
            return 8, 4264 + location_id - 1, minusd
        if variable == 't850' and 532 <= location_id <= 5328:
            # data9[0,    0] = (9, ' t850',  532) ## Not verified, last file is corrupt
            # data9[0, 4796] = (9, ' t850', 5328) ## Not verified, last file is corrupt
            return 9, 0 + location_id - 532, minusd

        raise ValueError("Provided column id does not exist")

    def __load_datafile(self, data_id):
        """Loads the requested datafile"""

        if self.__loaded == data_id:
            # Nothing to do
            return

        filename = "data_D_1980_2010_part%d.npy" % data_id
        filepath = os.path.join('data', filename)

        # clear memory
        self.__data = None

        # load new data
        self.__data = np.load(filepath)
        self.__loaded = data_id

        return

    def data_ready(self):
        """Check if data is ready to use. Returns boolean."""

        for i in range(10):
            filename = "data_D_1980_2010_part%d.npy" % i
            if not os.path.exists(os.path.join('data', filename)):
                return False

        return True

    def make_data(self):
        """Dummy method. Nothing to build, these files have to be manually
        downloaded"""
        return False

    def get_feature_col(self, column_id):
        """ Retrieves requested column

        This method takes a tuple as only argument. The tuple should be of size 3.
        column_id = (minusd, variable, location_id)
        data_source is the object holding loaded data

        Returns a dataframe with the requested feature column.

        The way to get the data is by reading the original column for a
        meteorological variable at specified location, and then shifting the index
        appropriately.
        """

        # unpack the input tuple
        data_id, col_n, minusd = self.__map(column_id)
        if data_id != self.__loaded:
            self.__load_datafile(data_id)

        # read partial dataframe
        col = pd.Series(self.__data[:, col_n].astype(float))
        col.index += 1

        col = col.loc[self.history_window - minusd
                      :11310 - self.predict_ahead - minusd] # should be 11309

        # fix index
        col.index += minusd

        return col

    def get_label_col(self):
        """Returns a dataframe with the labels column"""

        # if parameters match TA's, use TA's data for labels
        if self.history_window == 4 and self.predict_ahead == 7:
            # build the path name for the target file to read
            filepath = os.path.join('data', 'target_1980_2010.npy')

            target = pd.Series(np.load(filepath)[:, 1])
            target.index += self.history_window

            return target

        # else, use our own data that is complete

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

        # if parameters match TA's, use TA's data for labels
        if self.history_window == 4 and self.predict_ahead == 7:
            # build the path name for the target file to read
            filepath = os.path.join('data', 'target_1980_2010.npy')

            target = pd.Series(np.load(filepath)[:, 0])
            target.index += self.history_window

            return target

        # build the path name for the target file to read
        filepath = os.path.join('data', 'label', 'sum15.pickle')

        # read partial dataframe
        series = pd.read_pickle(filepath)
        series = series.loc[self.history_window + self.predict_ahead
                            :11310] # should be 11309
        series.index -= self.predict_ahead
        return series
