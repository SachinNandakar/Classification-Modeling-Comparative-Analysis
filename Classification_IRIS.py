from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

# Features of iris dataset
# Input features: 4 input features (numerical)
# print(iris.feature_names)
# Output feature: 1 output feature (categorical/factor)
# print(iris.target_names)

# View the dataset
# print(iris)
# Classification of dataset into two parts: data(4-variables) & target(1-variable)
# print(iris.data)
# print(iris.target)

# View the data dimension
# print(iris.data.shape)
# print(iris.target.shape)


# Build a classification model using Random Forest
##cls_rf = RandomForestClassifier()
##cls_rf.fit(iris.data, iris.target)
##
##print(cls_rf.predict(iris.data[[0]]))
##print(cls_rf.predict_proba(iris.data[[0]]))

# Data Split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.20)
# X_train.shape, Y_train.shape
# X_test.shape, Y_test.shape

res = []
bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, Y_train)
print("Bernoulli Classifier Score: ")
print(bnb.score(X_test, Y_test))
res.append(bnb.score(X_test, Y_test))

mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
print("Multinomial Classifier Score: ")
print(mnb.score(X_test, Y_test))
res.append(mnb.score(X_test, Y_test))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print("Gaussian Classifier Score: ")
print(gnb.score(X_test, Y_test))
res.append(gnb.score(X_test, Y_test))

lr = LogisticRegression()
lr.fit(X_train, Y_train)
print("Logistic Regression Score: ")
print(lr.score(X_test, Y_test))
res.append(lr.score(X_test, Y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
print("Decision Tree Classifier Score: ")
print(dt.score(X_test, Y_test))
res.append(dt.score(X_test, Y_test))

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
print("Random Forest Classifier Score: ")
print(clf.score(X_test, Y_test))
res.append(clf.score(X_test, Y_test))

ada = AdaBoostClassifier()
ada.fit(X_train, Y_train)
print("AdaBoost Classifier Score: ")
print(ada.score(X_test, Y_test))
res.append(ada.score(X_test, Y_test))

grd = GradientBoostingClassifier()
grd.fit(X_train, Y_train)
print("gradient Tree Boost Classifier Score: ")
print(grd.score(X_test, Y_test))
res.append(grd.score(X_test, Y_test))

print(np.array(res))

# Bar graph representation of performance score of different classification methods
fig = plt.figure(figsize = (12, 4)) 
ClsName = ['BernoulliNB', 'MultinomialNB', 'GaussianNB', 'Logistic_Reg', 'Decision_Tree',
         'Random_Forest', 'AdaBoost', 'Gradient_Tree']
plt.bar(ClsName,res)
plt.ylabel('Scores')
plt.title('Performance Scores of Classification Methods')
plt.show()

#####################################
# Graphical Representation of Accuracy Measurement of Decision Tree & Ensemble Methods
nb_classifications = 100
rf_accuracy = []

for i in range(1, nb_classifications):
    a = cross_val_score(RandomForestClassifier(n_estimators=i),
                        iris.data, iris.target, scoring = 'accuracy', cv=10).mean()
    rf_accuracy.append(a)

plt.plot(range(1,100), rf_accuracy)
plt.title('Random Forest Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()


ab_accuracy = []
for i in range(1, nb_classifications):
    a = cross_val_score(AdaBoostClassifier(n_estimators=i),
                        iris.data, iris.target, scoring = 'accuracy', cv=10).mean()
    ab_accuracy.append(a)

plt.plot(range(1,100), ab_accuracy)
plt.title('Ada Boost Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()


gt_accuracy = []
for i in range(1, nb_classifications):
    a = cross_val_score(GradientBoostingClassifier(n_estimators=i),
                        iris.data, iris.target, scoring = 'accuracy', cv=10).mean()
    gt_accuracy.append(a)

plt.plot(range(1,100), gt_accuracy)
plt.title('Gradient Tree Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()
