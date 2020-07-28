# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:48:50 2019

@author: Ali Mir
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd


#Load Data


dataset_trn = pd.read_csv('pd_speech_features2.csv')


X = dataset_trn.iloc[:, 0:752].values
y = dataset_trn.iloc[:, 752].values



"""Applying PCA"""

pca = decomposition.PCA(n_components=752)
pc = pca.fit_transform(X)
CoVar = pca.explained_variance_ratio_
CVSort = sorted(CoVar, reverse=True)
sum = 0.0
count = 0
component = 0
for items in CVSort:
    sum = sum + items
    count = count + 1
    if sum <= 0.999999999999990:
        component = count

pca = decomposition.PCA(n_components=103)
pca.fit(X)
X = pca.transform(X)
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 600)

#Training different models


MLP = MLPClassifier()
MLP.fit(X_train, y_train)
MLP_ACC = MLP.score(X_test, y_test)
#print("\nMLP: ", MLP_ACC)


SVM = SVC()
SVM.fit(X_train, y_train) 
SVM_ACC = SVM.score(X_test, y_test)
#print("\nSVM: ", SVM_ACC)



bagging = BaggingClassifier(SVC())
bagging.fit(X_train, y_train)
EN_ACC = bagging.score(X_test, y_test)



pipe_svm = Pipeline([('scl', StandardScaler()),
			('clf', SVC())])

pipe_MLP = Pipeline([('clf', MLPClassifier(max_iter=300))])

			
# Set grid search params
C = [1,2, 5, 10, 50, 100]
G = [0.001,0.01,0.1,1,10,100]


grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
		'clf__C': C,'clf__degree':[3,4,5],'clf__gamma': G}]


grid_params_MLP = [{'clf__activation':['identity','logistic','tanh','relu'],'clf__solver':['lbfgs','sgd','adam']}]

# Construct grid searches
jobs = 6

gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring='accuracy',
			cv=4,
			n_jobs=jobs)



gs_MLP = GridSearchCV(estimator=pipe_MLP,
			param_grid=grid_params_MLP,
			scoring='accuracy',
			cv=4,
			n_jobs=jobs)

# List of pipelines for ease of iteration
grids = [gs_svm, gs_MLP]

# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Support Vector Machine', 1: 'MLP'}

# Fit the grid search objects
if __name__ == "__main__":
    print('Performing model optimizations...')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
    	print('\nEstimator: %s' % grid_dict[idx])	
    	# Fit grid search	
    	gs.fit(X_train,y_train)
    	# Best params
    	print('Best params: %s' % gs.best_params_)
    	# Best training data accuracy
    	print('Best training accuracy: %.3f' % gs.best_score_)
    	# Predict on test data with best params
    	y_pred = gs.predict(X_test)
    	# Test data accuracy of model with best params
    	print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    	# Track best (highest test accuracy) model
    	if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
            output = gs.predict(X_test)
            with open('predictions.txt', 'w') as outfile:
                for items in output:
                    outfile.write(str(items) + '\n')
                outfile.close()
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
#    
#    # Save best grid search pipeline to file
#    dump_file = 'best_gs_pipeline.pkl'
#    joblib.dump(best_gs, dump_file, compress=1)
#    print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))