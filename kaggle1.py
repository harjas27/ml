import pandas as pd
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm,tree
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('train.csv')
df.drop(['PassengerId'], 1, inplace=True)
df.drop(['Cabin'], 1, inplace=True)
df.drop(['Ticket'], 1, inplace=True)
df.drop(['Name'], 1, inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].median())

df["Sex"][df["Sex"] == "male"] = 0
df["Sex"][df["Sex"] == "female"] = 1

df["Embarked"] = df["Embarked"].fillna("S")
df["Embarked"][df["Embarked"] == "S"] = 0
df["Embarked"][df["Embarked"] == "C"] = 1
df["Embarked"][df["Embarked"] == "Q"] = 2
df["Child"] = float('NaN')

df["Child"][df["Pclass"] == 3] = 1
df["Child"][(df["Pclass"] == 2)] = 0
df["Child"][(df["Pclass"] == 1)] = 0
df["Pclass"][df["Pclass"] == 2] = 1


df1 = pd.read_csv('test.csv')
PassengerId = np.array(df1["PassengerId"]).astype(int)
df1.drop(['PassengerId'], 1, inplace=True)
df1.drop(['Cabin'], 1, inplace=True)
df1.drop(['Ticket'], 1, inplace=True)
df1.drop(['Name'], 1, inplace=True)

df1['Age'] = df1['Age'].fillna(df1['Age'].median())
df1['Fare'] = df1['Fare'].fillna(df1['Fare'].median())

df1["Sex"][df1["Sex"] == "male"] = 0
df1["Sex"][df1["Sex"] == "female"] = 1

df1["Embarked"] = df1["Embarked"].fillna("S")
df1["Embarked"][df1["Embarked"] == "S"] = 0
df1["Embarked"][df1["Embarked"] == "C"] = 1
df1["Embarked"][df1["Embarked"] == "Q"] = 2
df1["Child"] = float('NaN')

df1["Child"][df1["Age"] < 18] = 1
df1["Child"][df1["Age"] >= 18] = 0

X = df[["Pclass","Sex","Fare"]].values
Y = np.array(df['Survived'])

print(df["Survived"][df["Pclass"] == 1].value_counts(normalize = True))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

#clf = neighbors.KNeighborsClassifier()
clf = tree.DecisionTreeClassifier(max_depth = 50, min_samples_split = 5, random_state = 1)
#clf = svm.SVC()
#clf = RandomForestClassifier(max_depth = 10, min_samples_split=5 ,n_estimators = 100, random_state = 1)

clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
#print(df.head())
print(clf.feature_importances_)
'''test_features = df1[["Pclass","Sex","Fare","Child"]].values
my_prediction = clf.predict(test_features)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
#print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_five.csv", index_label = ["PassengerId"])
#print("my_solution_one.csv")'''