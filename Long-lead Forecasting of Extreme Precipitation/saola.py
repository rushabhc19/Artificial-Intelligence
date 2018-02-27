#!/usr/bin/python
"""SAOLA Algorithm. This is a mockup, include more methods if needed"""
import os
import numpy as np
#from numpy_data import Data
from pickle_data import Data

def saola_algorithm():
    """The algorithm. I should return a list of tuples. Each tuple represents a
    single feature column.

    i.e.
    return [(4, 'pw', 4567), (3, 't850', 3456), ... , (2, 'z1000', 2345)]
    """

    data = Data()

    #label_col = data.get_label_col()
    label_col = data.get_sum15_col()

    variables = ['z300', 'v850', 'u300', 'z1000', 'u850', 'z500',
                 'pw', 'v300', 't850']

    # Holds the features that are related to the label
    # key = tuple like (4, 'pw', 4567)
    # val['zscore_w_label'] = float, is z-score for feature against label
    # val['pd_series'] = pandas.Series, the actual column
    feature_dict = {}

    # variables for progress indicator
    total_count = 0.0
    display_progress = 0
    max_count = 5328 * 9 * data.history_window
    print "\n"


    delta_1 = 1.96
    delta_2 = 1.96

    for var in variables:
        for location in range(1, 5328 + 1):
            for minusd in range(0, data.history_window):

                # update and display progress indicator
                total_count += 1
                if total_count > display_progress:
                    display_progress += 123
                    percent = ((total_count/max_count)*100)
                    print ("\033[FWorking on (%2d, %5s, %4d) "
                           "num of selected features %3d: [% 5.1f%%]"
                           % (minusd, var, location, len(feature_dict), percent))

                # this is the feature under consideration
                col_id = (minusd, var, location)
                feature_i = data.get_feature_col(col_id)

                # relation with feature_i to label (z-score)
                feature_rel_label = data.func_i(feature_i, label_col)
                if feature_rel_label < delta_1:
                    # ignore this one, go to next feature
                    continue
                else:
                    rel_indicator = True

                # compare related feature to the other related features
                for key, val in feature_dict.items():
                    # get column for feature_y
                    feature_y = val['pd_series']
                    # relation between feature_i to feature_y
                    feat_i_rel_feat_y = data.func_i(feature_i, feature_y)
                    # taken from SAOLA paper page 663
                    #delta_2 = min(feature_rel_label, val['zscore_w_label'])
                    if (val['zscore_w_label'] > feature_rel_label
                            and feat_i_rel_feat_y >= delta_2):
                        # DO NOT ADD feature_i to set of features
                        rel_indicator = False
                        break
                    if (feature_rel_label > val['zscore_w_label']
                            and feat_i_rel_feat_y >= delta_2):
                        # feature_i is better related to label than feature_y
                        # delete feature_y
                        feature_dict.pop(key)

                if rel_indicator:
                    # add current feature to dict
                    feature_dict[col_id] = {
                        'zscore_w_label': feature_rel_label,
                        'pd_series': feature_i}

    print ("\033[F%6d features have been processed.%s\n\n"
           "List of selected features:\n"
           % (total_count, " " * 35))

    return feature_dict

def main():
    """The main method"""
    selected_features = saola_algorithm()
    print selected_features.keys()

    np.save(os.path.join('data', 'selected_features.npy'), selected_features)

if __name__ == "__main__":
    main()
