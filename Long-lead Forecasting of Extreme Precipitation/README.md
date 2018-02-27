# README #

This repository contains code related the AI Team project for Team #3

### What is this repository for? ###

* To maintain collaboration with teammates
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

#### 1. Make sure you have the raw data files

You will need the folder named `raw_data` which contains all the necessary files to begin with.

#### 2. Combine raw variable files to contain data for every available year

For this you use the bash script named `aggregate_csvs.bash`

The generated files will occupy 3GB of storage

```
 $ bash aggregate_csvs.bash 
raw_data/iowa_1980.csv >> raw_data/aggreg/iowa.csv
raw_data/iowa_1981.csv >> raw_data/aggreg/iowa.csv
raw_data/iowa_1982.csv >> raw_data/aggreg/iowa.csv
.
.
.
raw_data/z500_2008.csv >> raw_data/aggreg/z500.csv
raw_data/z500_2009.csv >> raw_data/aggreg/z500.csv
raw_data/z500_2010.csv >> raw_data/aggreg/z500.csv
 $ ls raw_data/aggreg
iowa.csv  pw.csv  t850.csv  u300.csv  u850.csv  v300.csv  v850.csv  z1000.csv  z300.csv  z500.csv
```
#### 3. Generate usable data files

You can use a python or an IPython session to build the files that are necessary for processes that we want to complete.


```
 $ ipython
Python 2.7.12 (default, Nov 20 2017, 18:23:56) 
Type "copyright", "credits" or "license" for more information.

IPython 5.5.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: from pickle_data import Data

In [2]: data = Data()

In [3]: data.write_col_pickles()
Reading variable pw...
Writing pickle for location_id   5328
Reading variable t850...
Writing pickle for location_id   5328
Reading variable u300...
Writing pickle for location_id   5328
Reading variable u850...
Writing pickle for location_id   5328
Reading variable v300...
Writing pickle for location_id   5328
Reading variable v850...
Writing pickle for location_id   5328
Reading variable z1000...
Writing pickle for location_id   5328
Reading variable z300...
Writing pickle for location_id   5328
Reading variable z500...
Writing pickle for location_id   5328
reading Iowa data...
Process complete!!!
```
You have now access to every column and the class labels. This is how you'd read these:
```
In [4]: target_sum15 = data.get_sum15_col()

In [5]: target_label = data.get_label_col()

In [6]: col_id_a = (3, 'pw', 5328)

In [7]: feature_a = data.get_feature_col(col_id_a)

In [8]: col_id_b = (0, 'v300', 500)

In [9]: feature_b = data.get_feature_col(col_id_b)

```
And now calculate pearson correlations and z-scores between pairs:
```
In [10]: feature_a.corr(feature_b)
Out[10]: -0.0027685928330419478

In [11]: feature_b.corr(target_label)
Out[11]: 0.019918570637155568

In [12]: feature_b.corr(target_sum15)
Out[12]: 0.056757257081995083

In [13]: data.func_i(feature_a, feature_b)
Out[13]: 0.29426713728867265

In [14]: data.func_i(feature_b, target_sum15)
Out[14]: 6.039068547041178
```

These methods can now be used in the SAOLA algorithm implementation.
