import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import KFold

def load_data(filename, skiprows = 1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.

    Inputs:
        filename: given as a string.

    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=skiprows, delimiter=',')

def classification_err(y, real_y):
    error = 0
    for i in range(len(y)):
        if (y[i] != real_y[i]):
            error += 1.0
    return error / float(len(y))

training_data = load_data('train_2008.csv')
test_data = load_data('test_2008.csv')


# Get rid of first 3 columns : HHID not necessary for ML, neither month or year of survey
# Columns 0-11 regular data, 12 weights, potential distribution?, 13-19 regular data, 20 HRHHID2 not necessary?
# 29-30 binary? 35 binary?
X_train = training_data[:,3:-1]
Y_train = training_data[:,-1]

X_test = test_data[:,3:]

# seed = np.random.get_state()
kf = KFold(n_splits = 5, shuffle=True)


######################## Random Forest Model #########################
def run_random_forest(X_train, Y_train):
    total_error = 0
    clf = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', n_jobs=-1)
    for train_index, test_index in kf.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        clf = clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        total_error += classification_err(predict, y_test)
    print(total_error/float(5))

run_random_forest(X_train, Y_train)




######################### Neural Network Model ########################
def run_neural_net(X_train, Y_train):
    order = np.random.permutation(len(X_train))
    y_train = keras.utils.np_utils.to_categorical(Y_train)

    model = Sequential()
    model.add(Dense(334, input_shape=(379,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(333))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(333))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # model.summary()
    X_trainNN = X_train[:]
    for i in range(len(X_trainNN[0])):
        if (max(X_trainNN[:,i]) != 0):
            X_trainNN[:,i]/max(X_trainNN[:,i])
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop', metrics=['accuracy'])

    fit = model.fit(X_train[order], y_train[order], batch_size=600, epochs=20)

    score1 = model.evaluate(X_train[:40000], y_train[:40000], verbose=0)
    score2 = model.evaluate(X_train[40000:], y_train[40000:], verbose=0)
    print('Train accuracy:', score1[1])
    print('Test accuracy:', score2[1])

#################################### SVM ##############################

# run_neural_net(X_train, Y_train)


# Prints out probability predictions into an excel file for Kaggle submission.
f = open("submission.csv", "w")
predict = clf.predict_proba(X_test)
f.write("id,target\n")
for i in range(len(predict)):
    f.write('{},{}\n'.format(i, predict[i][1]))
f.close()
