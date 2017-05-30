import numpy as np
from sklearn import preprocessing, cross_validation, neighbors,svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['1000025'], 1, inplace=True)

#print(list(df))

#defining features and target
X = np.array(df.drop(['2.1'],1))
Y = np.array(df['2.1'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = svm.SVC()

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
'''example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
print(df.shape)
print(X.shape)
print(X_train.shape)
print(len(example_measures))'''