###TASK:Get a head start on how to use existing machine learning libraries to do classification on iris dataset###

from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as pyt

iris =load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))

pyt.plot(k_range, scores)
pyt.xlabel('Value of K for KNN')
pyt.ylabel('Testing Accuracy')
pyt.show()
