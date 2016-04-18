# scikit-viz

scikit-viz aims to provide a collection of templates to make producing plots to evaluate models built using scikit-learn much simpler. 

### Installation
```sh
$ pip install scikit-viz
```

### Usage
```
from sklearnviz import plot_roc
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
y[y==2] = 0
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X,y)
y_pred = clf.predict_proba(X)[:,1]

# this is where the magic happens
plot_roc(y, y_pred, spacing=0.1)
```

![roc_curve](https://raw.githubusercontent.com/fernavid/scikit-viz/master/tests/roc_example.png)
