from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import helper_func as hp



## Decision Tree Classifier - Iris Example 
## Modified from scikit-learn tutorials
def dtIrisExample():
	iris = load_iris()
	X = iris.data
	y = iris.target
	X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=5)
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train,y_train)
	pred = clf.predict(X_test)
	print("Accuracy: %.2f" % metrics.accuracy_score(pred,y_test))
	hp.saveTree(clf,"iris_dt.pkl")
	clf = hp.loadTree("iris_dt.pkl")
	hp.visualizeTree(clf,iris.feature_names,iris.target_names,"Iris")
	return None

if __name__=="__main__":
	dtIrisExample()