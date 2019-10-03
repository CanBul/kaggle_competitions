import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['Cabin'].fillna('U', inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])

test_data['Cabin'].fillna('U', inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
replacement = {
    'T': 0.0,
    'U': 1.0,
    'A': 2.0,
    'G': 3.0,
    'C': 4.0,
    'F': 5.0,
    'B': 6.0,
    'E': 7.0,
    'D': 8.0
}

train_data['Cabin'] = train_data['Cabin'].apply(lambda x: replacement.get(x))
train_data['Cabin'] = StandardScaler().fit_transform(train_data['Cabin'].values.reshape(-1, 1))

test_data['Cabin'] = test_data['Cabin'].apply(lambda x: replacement.get(x))
test_data['Cabin'] = StandardScaler().fit_transform(test_data['Cabin'].values.reshape(-1, 1))

train_data['Title']= train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title']= test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

test_data=test_data.drop(['Name','Ticket', 'PassengerId'], axis=1)
train_data=train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1)


train_data['Family']=train_data['SibSp']+ train_data['Parch']+1
test_data['Family'] = test_data['SibSp']+test_data['Parch']+1
train_data['Isalone'] = 'Yes'
test_data['Isalone'] = 'Yes'
train_data['Isalone'].loc[train_data['Family']>1] = 'No'
test_data['Isalone'].loc[test_data['Family']>1] = 'No'


train_data['Age'] = train_data.apply(
    lambda row: 4.57 if np.isnan(row['Age']) and row["Title"]=="Master" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 21.84 if np.isnan(row['Age']) and row["Title"]=="Miss" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 32.36 if np.isnan(row['Age']) and row["Title"]=="Mr" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 35.78 if np.isnan(row['Age']) and row["Title"]=="Mrs" else row['Age'],   axis=1 )
train_data['Age'] = train_data.apply(
    lambda row: 45.54 if np.isnan(row['Age']) and row["Title"]=="Other" else row['Age'],   axis=1 )

test_data['Age'] = test_data.apply(
    lambda row: 4.57 if np.isnan(row['Age']) and row["Title"]=="Master" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 21.84 if np.isnan(row['Age']) and row["Title"]=="Miss" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 32.36 if np.isnan(row['Age']) and row["Title"]=="Mr" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 35.78 if np.isnan(row['Age']) and row["Title"]=="Mrs" else row['Age'],   axis=1 )
test_data['Age'] = test_data.apply(
    lambda row: 45.54 if np.isnan(row['Age']) and row["Title"]=="Other" else row['Age'],   axis=1 )

test_data['Fare'] = test_data['Fare'].fillna(10)
train_data['Fare'] = train_data['Fare'].fillna(10)

train_data['Fare'] = StandardScaler().fit_transform(train_data['Fare'].values.reshape(-1, 1))
train_data['Age'] = StandardScaler().fit_transform(train_data['Age'].values.reshape(-1, 1))

test_data['Fare'] = StandardScaler().fit_transform(test_data['Fare'].values.reshape(-1, 1))
test_data['Age'] = StandardScaler().fit_transform(test_data['Age'].values.reshape(-1, 1))

data1_x = ['Sex','Pclass', 'Embarked', 'SibSp', 'Parch', 'Title', 'Family', 'Isalone','Age', 'Fare', 'Cabin' ]
X_train = pd.get_dummies(train_data[data1_x])
X_test = pd.get_dummies(test_data[data1_x])
Y_train = train_data['Survived']

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_nb = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_lsvm = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dtree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rfor = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
acc_mlp

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Stochastic Gradient Descent',  
              'Linear SVC', 
              'Decision Tree', 'Neural Net'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,  
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_mlp]})
models.sort_values(by='Score', ascending=False)
