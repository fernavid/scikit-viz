# scikit-viz

scikit-viz aims to provide a collection of templates to make producing plots to evaluate models built using scikit-learn much simpler. 

### Installation
```sh
$ pip install scikit-viz
```

### Current Features
 - __plot_roc__(y, y_pred, spacing=0.2, indifference=None)
    - __y__ actual value (1 or 0)
    - __y_pred__ probability of 1
    - __spacing__ governs the spacing between threshold labels
    - __indifference__ is a dictionary defined as follows:
        ```python
        {
            '__rate_positive__': 0.95, # rate of 1 case
            '__rate_negative__': 0.05, # rate of 0 case
            '__tp_util__': 20, # utility of true positives
            '__tn_util__': -50, # utility of true negatives
            '__fp_util__': -300, # utility of false positives
            '__fn_util__': -50 # utility of false negatives
        }
        ```
        Provide this dictionary describing the economic utility and frequency of false positives and false negatives, and it will produce an __indifference curve__ on the plot which provides a threshold above which the classifier is economically useful. For more information on this extremely interesting concept, see this blog post (http://blog.mldb.ai/blog/posts/2016/01/ml-meets-economics/)
    

### Usage
```python
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
