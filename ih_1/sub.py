import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation,tree,svm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop(['Id'],1,inplace=True)
train['DetectedCamera'][train['DetectedCamera'] == 'Rear'] = 1
train['DetectedCamera'][train['DetectedCamera'] == 'Left'] = 2
train['DetectedCamera'][train['DetectedCamera'] == 'Right'] = 3
train['DetectedCamera'][train['DetectedCamera'] == 'Front'] = 4

train['SignFacing'][train['SignFacing'] == 'Rear'] = 1
train['SignFacing'][train['SignFacing'] == 'Left'] = 2
train['SignFacing'][train['SignFacing'] == 'Right'] = 3
train['SignFacing'][train['SignFacing'] == 'Front'] = 4
train.dropna(inplace=True)

X = train[['AngleOfSign']].values
Y = train['SignFacing'].values

print(train['SignFacing'].value_counts(normalize = True))
print(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

#clf = tree.DecisionTreeClassifier(max_depth = 50, min_samples_split = 5, random_state = 1)
clf=svm.SVC()

clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
print(clf.feature_importances_)