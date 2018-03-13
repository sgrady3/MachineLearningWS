from sklearn import tree
from sklearn.externals import joblib
import graphviz



## Visualization option for tree
def visualizeTree(clf,features,classes,render_name = "Decision Tree"):
	dot_data = tree.export_graphviz(clf,out_file=None, feature_names=features,  \
                         class_names=classes,  \
                         filled=True, rounded=True,  \
                         special_characters=True) 
	graph = graphviz.Source(dot_data)
	graph.render(render_name,cleanup=True)

## Simple function to save a tree. Scikit's joblib is more efficient 
## when handling a large number of np.arrays than Python's pickle library
def saveTree(clf,fileName):
	joblib.dump(clf,fileName)
	return None

## Simple function to load a tree from a file.
def loadTree(fileName):
	return joblib.load(fileName)
