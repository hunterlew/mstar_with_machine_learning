from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def dt(criterion="entropy", max_features="sqrt"):
    return DecisionTreeClassifier(criterion=criterion, max_features=max_features, max_depth=None, random_state=0)

def rf(n_tree=100, max_features="sqrt"):
    return RandomForestClassifier(n_estimators=n_tree, max_features=max_features, min_samples_split=2, \
                                  max_depth=None, bootstrap=True, oob_score=False, random_state=0, n_jobs=4)

def gbdt(n_tree=100, max_features="sqrt"):
    return GradientBoostingClassifier(n_estimators=n_tree, learning_rate=0.005, \
                                      max_features=max_features, max_depth=None, random_state=0)

def logit(C=1.0):
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)

def mlp(hidden=(100), act="logistic", batch=32):
    return MLPClassifier(hidden_layer_sizes=hidden, activation=act, solver="sgd", batch_size=batch, \
                         learning_rate="constant", learning_rate_init=0.1, early_stopping=False, max_iter=1000, random_state=0)

def svm(C=1.0, kernel="rbf"):
    return SVC(C=C, kernel=kernel, max_iter=-1, random_state=0)

def knn(n_neighbors=10, weights="distance"):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm="auto")

def bayes():
    return GaussianNB()

def train(X, y, classifier):
    return classifier.fit(X, y)

def test(X, classifier):
    return classifier.predict(X)

def acc(X, y, classifier):
    return classifier.score(X, y)
