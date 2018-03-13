import pandas
import random
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

##load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

##split data set into train and test datasets

random.seed(77)
X= array[:,0:8]
Y=array[:,8]
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


##AdaBoost
##http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

##define parameter distribution
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight={ 0:0.5, 1:0.5 })
param_dist = {
 'n_estimators': [50, 300],
 'learning_rate' : [0.01,0.05,0.1,0.2,0.3,0.5,1.0,2.0],
 'algorithm' : ['SAMME', 'SAMME.R']
 }

##tuning ababoost parameters using cross validation
clf = RandomizedSearchCV(AdaBoostClassifier(base_estimator=dt),param_distributions = param_dist,cv=3,n_iter = 10,n_jobs=1)

##fit training data
clf.fit(X_train,Y_train)
y_pred=clf.predict(x_test)
y_pred_prob=clf.predict_proba(x_test)

##evaluate model
print(metrics.accuracy_score(y_test, y_pred))
##accurary 0.7662337662337663

cfs = metrics.confusion_matrix(y_test, y_pred)
print(cfs)
##[[84 13]
 ##[23 34]]
TP = cfs[1, 1]
TN = cfs[0, 0]
FP = cfs[0, 1]
FN = cfs[1, 0]

##sensitivity 0.5964912280701754
print(metrics.recall_score(y_test, y_pred))

##specificity 0.865979381443299
specificity = TN / (TN + FP)
print(specificity)

##precision 0.723404255319149
print(metrics.precision_score(y_test, y_pred))

##plot ROC curve
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for cancer classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

##get AUC score 0.7312353047567373
print(metrics.roc_auc_score(y_test, y_pred_prob[:,1]))

##rerun the training fit by changing class weight
##change class_weight
dt = DecisionTreeClassifier(class_weight='balanced')
clf = RandomizedSearchCV(AdaBoostClassifier(base_estimator=dt),param_distributions = param_dist,cv=3,n_iter = 10,n_jobs=1)
clf.fit(X_train,Y_train)
y_pred=clf.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
##accuracy 0.7662337662337663

cfs = metrics.confusion_matrix(y_test, y_pred)
print(cfs)
##[[80 17]
 ##[19 38]]
TP = cfs[1, 1]
TN = cfs[0, 0]
FP = cfs[0, 1]
FN = cfs[1, 0]

##sensitivity 0.6666666666666666
print(metrics.recall_score(y_test, y_pred))

##specificity 0.8247422680412371
specificity = TN / (TN + FP)
print(specificity)

##precision 0.6909090909090909
print(metrics.precision_score(y_test, y_pred))

##get AUC score 0.7312353047567373
print(metrics.roc_auc_score(y_test, y_pred_prob[:,1]))


##exam breast cancer dataset
dataframe = pandas.read_csv('filter_set_gcrma.csv')
array = dataframe.values

##split data set into train and test datasets
random.seed(77)
X= array[:,0:1256] ##1256 variable genes
## X= array[:,1256:1318] ##62 PCAs
## X= array[:,1318:1321] ##3d tsne
Y=array[:,1321] ##labels
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

dt = DecisionTreeClassifier(class_weight='balanced')
param_dist = {
 'n_estimators': [50, 300],
 'learning_rate' : [0.01,0.05,0.1,0.2,0.3,0.5,1.0,2.0],
 'algorithm' : ['SAMME', 'SAMME.R']
 }
clf = RandomizedSearchCV(AdaBoostClassifier(base_estimator=dt),param_distributions = param_dist,cv=3,n_iter = 10,n_jobs=1)
clf.fit(X_train,Y_train)
y_pred=clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
##1256 variable genes as features to train. accuracy 0.810344827586
##62 PCs as features to train. accuracy 0.775862068966
##Top 2 PCs as features to train. accuracy 0.793103448276
##3 dimensions of tSNE as features to train. accuracy 0.793103448276
##2 dimensions tSNE as features to train. accuracy 0.775862068966

##Gradient Boosting
##http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
##from sklearn.ensemble import GradientBoostingClassifier
##from sklearn.ensemble import GradientBoostingRegressor

##clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=2)
##clf.fit(x_train, y_train)
##y_pred=clf.predict(x_test)
##y_pred_prob=clf.predict_proba(x_test)
##print(metrics.accuracy_score(y_test, y_pred))

##XGBoost
##http://xgboost.readthedocs.io/en/latest/python/python_intro.html
##from xgboost import XGBClassifier

##model = XGBClassifier()
##model.fit(x_train, y_train)

##make predictions for test data
##y_pred = model.predict(x_test)
##predictions = [round(value) for value in y_pred]

##evaluate predictions
##accuracy = metrics.accuracy_score(y_test, predictions)
##print("Accuracy: %.2f%%" % (accuracy * 100.0))
