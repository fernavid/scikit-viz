import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearnviz import plot_roc

iris = datasets.load_iris()
X = iris.data
y = iris.target
y[y==2] = 0

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X,y)

y_pred = clf.predict_proba(X)[:,1]

plot_roc(y, y_pred, spacing=0.1)