from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
import sys
from numpy import mean


def dtKFoldCrossVal(cv):
	iris = load_iris()
	X = iris.data
	y = iris.target
	X_train,X_test,y_train,y_test = train_test_split(X,y)
	clf = tree.DecisionTreeClassifier()
	scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
	return scores

if __name__ == "__main__":
	cv = 10
	if len(sys.argv) > 1:
		cv = int(sys.argv[1])
	scores = dtKFoldCrossVal(cv)
	print("Average accuracy score from %d-fold cross validation: %.2f"%(cv,mean(scores)))