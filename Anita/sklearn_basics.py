"""
imports
"""
import argparse
import pandas as pd
import numpy as np
import random as rnd
#import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

"""
arguments
"""
parser=argparse.ArgumentParser()
parser.add_argument('--split', help='train/test split', default=0.9,type=int)
args = parser.parse_args()

"""
dataframes
"""
path_to_pos = '../LSTM/pos_sample.csv'
path_to_neg = '../LSTM/neg_sample.csv'
path_to_neg_test = '../LSTM/neg_test_sample.csv'

pos = pd.read_csv(path_to_pos)
neg = pd.read_csv(path_to_neg)
neg_test = pd.read_csv(path_to_neg_test)

frames = [pos, neg]
data = pd.concat(frames)
data = data.sample(frac=1)  #shuffle using sample, frac=1
#print data.info()

#use iloc for splitting into train/test sets
train_df = data.iloc[0:int(data.shape[0]*args.split)]
test_df = data.iloc[int(data.shape[0]*args.split):]
#print test_df.describe()

train_data = train_df.drop('is_attributed', axis=1)
train_label = train_df['is_attributed']
test_data = test_df.drop('is_attributed', axis=1).copy()
test_label = test_df['is_attributed']
#print test_data.describe()
#exit()

"""
logistic regression
"""
logreg = LogisticRegression()
logreg.fit(train_data, train_label)
test_pred = logreg.predict(test_data)
test_pred_neg = logreg.predict(test_data)
acc_log = round(logreg.score(train_data, train_label) * 100, 2)
roc_auc_log = roc_auc_score(test_label,test_pred)
logloss_log = log_loss(test_label, test_pred)
print 'logistic regression acc: ',acc_log
print 'logistic regression auc: ',roc_auc_log
print 'logistic regression logloss: ',logloss_log
print
#exit()
#coeff_df = pd.DataFrame(train_df.columns.delete(0))
#coeff_df.columns = ['Feature']
#coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

#print  coeff_df.sort_values(by='Correlation', ascending=False)
"""
support vector machine

svc = SVC()
svc.fit(train_data, train_label)
test_pred = svc.predict(test_data)
acc_svc = round(svc.score(train_data, train_label) * 100, 2)
roc_auc_svc = roc_auc_score(test_label,test_pred)
logloss_svc = log_loss(test_label,test_pred)
print 'support vector machine acc: ',acc_svc
print 'support vector machine auc: ',roc_auc_svc
print 'support vector machine logloss: ',logloss_svc
"""
"""
k nearest neighbors
"""
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_data, train_label)
test_pred = knn.predict(test_data)
acc_knn = round(knn.score(train_data, train_label) * 100, 2)
roc_auc_knn = roc_auc_score(test_label,test_pred)
logloss_knn = log_loss(test_label,test_pred)
print 'k nearest neighbors acc: ',acc_knn
print 'k nearest neighbors auc: ',roc_auc_knn
print 'k nearest neighbors logloss: ',logloss_knn
print

"""
gaussian naive bayes
"""
gaussian = GaussianNB()
gaussian.fit(train_data, train_label)
test_pred = gaussian.predict(test_data)
acc_gaussian = round(gaussian.score(train_data, train_label) * 100, 2)
roc_auc_gaussian = roc_auc_score(test_label,test_pred)
logloss_gaussian = log_loss(test_label,test_pred)
print 'gaussian naive bayes acc: ',acc_gaussian
print 'gaussian naive bayes auc: ',roc_auc_gaussian
print 'gaussian naive bayes logloss: ',logloss_gaussian
print

"""
perceptron
"""
perceptron = Perceptron()
perceptron.fit(train_data, train_label)
test_pred = perceptron.predict(test_data)
acc_perceptron = round(perceptron.score(train_data, train_label) * 100, 2)
roc_auc_perceptron = roc_auc_score(test_label,test_pred)
logloss_perceptron = log_loss(test_label,test_pred)
print 'perceptron acc: ',acc_perceptron
print 'perceptron auc: ',roc_auc_perceptron
print 'perceptron logloss: ',logloss_perceptron
print

"""
linear svc

linear_svc = LinearSVC()
linear_svc.fit(train_data, train_label)
test_pred = linear_svc.predict(test_data)
acc_linear_svc = round(linear_svc.score(train_data, train_label) * 100, 2)
roc_auc_svc = roc_auc_score(test_label,test_pred)
logloss_svc = log_loss(test_label,test_pred)
print 'linear svc acc: ',acc_linear_svc
print 'linear svc auc: ',roc_auc_linear_svc
print 'linear svc logloss: ',logloss_linear_svc
print
"""
"""
stochaic gradient decent

sgd = SGDClassifier()
sgd.fit(train_data, train_label)
test_pred = sgd.predict(test_data)
acc_sgd = round(sgd.score(train_data, train_label) * 100, 2)
roc_auc_sgd = roc_auc_score(test_label,test_pred)
logloss_sgd = log_loss(test_label,test_pred)
print 'stochaic gradient decent acc: ',acc_sgd
print 'stochaic gradient decent auc: ',roc_auc_sgd
print 'stochaic gradient decent logloss: ',logloss_sgd
print
"""
"""
decision tree classifier
"""
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, train_label)
test_pred = decision_tree.predict(test_data)
acc_decision_tree = round(decision_tree.score(train_data, train_label) * 100, 2)
roc_auc_decision_tree = roc_auc_score(test_label,test_pred)
logloss_decision_tree = log_loss(test_label,test_pred)
print 'decision tree classifier acc: ',acc_decision_tree
print 'decision tree classifier auc: ',roc_auc_decision_tree
print 'decision tree classifier logloss: ',logloss_decision_tree
print

"""
random forest
"""
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_data, train_label)
test_pred = random_forest.predict(test_data)
random_forest.score(train_data, train_label)
acc_random_forest = round(random_forest.score(train_data, train_label) * 100, 2)
roc_auc_random_forest = roc_auc_score(test_label,test_pred)
logloss_random_forest = log_loss(test_label,test_pred)
print 'random forest acc: ',acc_random_forest
print 'random forest auc: ',roc_auc_random_forest
print 'random forest logloss: ',logloss_random_forest
print
