


from sklearn import datasets,metrics
iris=datasets.load_iris()
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(max_depth=3)
classifier.fit(iris.data,iris.target)
iris_pred=classifier.predict(iris.data)
score=metrics.r2_score(iris.target,iris_pred)
print('Accuracy using decision tree classifier = ',score)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(3)
classifier.fit(iris.data,iris.target)
iris_pred=classifier.predict(iris.data)
score=metrics.r2_score(iris.target,iris_pred)
print('Accuracy using KNeighbors classifier = ',score)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
classifier.fit(iris.data,iris.target)
iris_pred=classifier.predict(iris.data)
score=metrics.r2_score(iris.target,iris_pred)
print('Accuracy using Random Forest classifier = ',score)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(iris.data,iris.target)
iris_pred=classifier.predict(iris.data)
score=metrics.r2_score(iris.target,iris_pred)
print('Accuracy using naive bayes classifier = ',score)


