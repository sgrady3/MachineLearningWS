from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np 
import matplotlib.pyplot as plt 

## An example showing how the size of the training set can affect the performance
## of the decision tree. Illustrates the concept of overfitting a model. 
def dtTrainingSize():
	iris = load_iris()
	X = iris.data
	y = iris.target
	clf = tree.DecisionTreeClassifier()
	scores = []
	train_size = np.arange(1,125,1)
	for i in train_size:
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=7)
		clf = clf.fit(X_train,y_train)
		pred = clf.predict(X_test)
		scores.append(metrics.accuracy_score(pred,y_test))
	plt.plot(train_size,scores,"-")
	plt.title('Decision Tree Performance:\nVarying size of Training Data')
	plt.xlabel('Training Set Size')
	plt.ylabel('Accuracy')
	plt.show()
	return None

if __name__ == "__main__":
	dtTrainingSize()