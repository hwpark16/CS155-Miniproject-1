import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def load_data(filename, skiprows = 1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.

    Inputs:
        filename: given as a string.

    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=skiprows, delimiter=',')

training_data = load_data('train_2008.csv')
test_data_2008 = load_data('test_2008.csv')
test_data_2012 = load_data('test_2012.csv')


# Get rid of first 3 columns : HHID not necessary for ML, neither month or year of survey
# Columns 0-11 regular data, 12 weights, potential distribution?, 13-19 regular data, 20 HRHHID2 not necessary?
# 29-30 binary? 35 binary?
X_train = training_data[:,3:-1]
Y_train = training_data[:,-1]

X_test_2008 = test_data_2008[:,3:]
X_test_2012 = test_data_2012[:,3:]

# Normalizes data
for i in range(len(X_train[0])):
    if (max(X_train[:,i]) != 0):
         X_train[:,i] = X_train[:,i]/max(X_train[:,i])


seed = np.random.get_state()
kf = KFold(n_splits = 5, shuffle=True)


######################## Random Forest Model #########################
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(1200, 2500, 10)]
# Number of features to consider at every split
max_features = ['auto', 'log2']
max_features.append(None)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criterion = ["gini", "entropy"]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion' : criterion}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(n_jobs = -1, n_estimators = 1900, min_samples_leaf = 3,
min_samples_split = 5, max_features = None, bootstrap= True, criterion= "entropy",
max_depth = None)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, scoring="roc_auc",
 param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf.fit(X_train, Y_train)
rf_random.best_params_

prediction = rf.predict(X_train[2000:5000])
roc_auc_score(Y_train[2000:5000], prediction)


# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [45, 50, 53],
    'max_features': [None],
    'min_samples_leaf': [3],
    'min_samples_split': [5],
    'n_estimators': [1890, 1920, 1950],
    'criterion': ["entropy"]
}
# Create a based model
rf = RandomForestClassifier(n_jobs=-1)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, scoring="roc_auc", param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, Y_train)
grid_search.best_params_
######################### Neural Network Model ########################
def run_neural_net(X_train, Y_train):
    print('helo1')
    order = np.random.permutation(len(X_train))
    y_train = keras.utils.np_utils.to_categorical(Y_train)

    model = Sequential()
    model.add(Dense(1000, input_shape=(379,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='RMSprop', metrics=['auc'])

    print('helo2')
    fit = model.fit(X_train[order], y_train[order], batch_size=600, epochs=10)
    print('helo3')
    score1 = model.evaluate(X_train[:40000], y_train[:40000], verbose=0)
    score2 = model.evaluate(X_train[40000:], y_train[40000:], verbose=0)
    print('Train accuracy:', score1[1])
    print('Test accuracy:', score2[1])

############################################################################################

run_neural_net(X_train, Y_train)


# Prints out probability predictions into an excel file for Kaggle submission.
f = open("submission.csv", "w")
predict = rf_random.predict_proba(X_test_2008)
f.write("id,target\n")
for i in range(len(predict)):
    f.write('{},{}\n'.format(i, predict[i][1]))
f.close()

f = open("submission_2012.csv", "w")
predict = rf_random.predict_proba(X_test_2012)
f.write("id,target\n")
for i in range(len(predict)):
    f.write('{},{}\n'.format(i, predict[i][1]))
f.close()
