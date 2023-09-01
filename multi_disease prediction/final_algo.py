import pandas as pd
import numpy as np
# Libraries for ML model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import classi_rprt

# Reading the training Dataset
train_df = pd.read_csv("Training.csv")
print(train_df.head())
print(train_df['prognosis'].unique())
print(train_df['prognosis'].nunique())

# Reading the testing Dataset
test_df = pd.read_csv("Testing.csv")
test_df.head()


train_df.drop('Unnamed: 133', axis=1, inplace=True)
train_df.info()


# split dataset into attributes and labels
X_train = train_df.iloc[:,:-1].values # the training attributes
y_train = train_df.iloc[:,132].values # the training labels
X_test = test_df.iloc[:,:-1].values # the testing attributes
y_test = test_df.iloc[:,132].values # the testing labels

target_names_ = train_df.prognosis.unique()
target_names=[]
for targets in target_names_:
    if ' ' in targets:
        target_names.append(targets.replace(' ', ''))
    else:
        target_names.append(targets)

#models
print("=========================================================================")
print("==========================Decision Tree=================================")
print("=========================================================================")
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_predDT = decision_tree.predict(X_test)
acc_dt = round(accuracy_score(y_train, decision_tree.predict(X_train))*100,2)
print("Decision Tree Train Accuracy: ",round(accuracy_score(y_train, decision_tree.predict(X_train))*100,2))
print("Decision Tree Test Accuracy: ",round(accuracy_score(y_test, Y_predDT) * 100 ,2))
print("==========================Confusion matrix=================================")
matrix =confusion_matrix(y_test, Y_predDT)
class_names=target_names
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("==========================Classification Report=================================")
print(classification_report(y_test, Y_predDT, target_names=target_names))
rprt = classification_report(y_test, Y_predDT, target_names=target_names)
classi_rprt.plot_classification_report(rprt)


print("=========================================================================")
print("==========================Random Forest==================================")
print("=========================================================================")
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_rf = round(accuracy_score(y_train, decision_tree.predict(X_train))*100,2)
print("Random Forest Train Accuracy: ",round(accuracy_score(y_train, random_forest.predict(X_train))*100,2))
print("Random Forest Test Accuracy: ",accuracy_score(y_test, Y_predRF)*100)
print("==========================Confusion matrix=================================")
matrix =confusion_matrix(y_test, Y_predRF)
class_names=target_names
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("==========================Classification Report=================================")
print(classification_report(y_test, Y_predRF, target_names=target_names))
rprt = classification_report(y_test, Y_predRF, target_names=target_names)
classi_rprt.plot_classification_report(rprt)

print("=========================================================================")
print("========================Voting Classifier================================")
print("=========================================================================")
#Voting Classifier
logreg = LogisticRegression()
random_forest = RandomForestClassifier(n_estimators=100)
estimators=[('Logistic', logreg),('RandomForest', random_forest)]
vc = VotingClassifier(estimators,voting='soft')  
vc.fit(X_train,y_train)
Y_predVC = vc.predict(X_test)
acc_vc = round(accuracy_score(y_train, decision_tree.predict(X_train))*100,2)
print("Voting Classifier Train Accuracy: ",round(accuracy_score(y_train, vc.predict(X_train))*100,2))
print("Voting Classifier Test Accuracy: ",round(accuracy_score(y_test, Y_predVC) * 100,2))
print("==========================Confusion matrix=================================")
matrix =confusion_matrix(y_test, Y_predVC)
class_names=target_names
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
print("==========================Classification Report=================================")
print(classification_report(y_test, Y_predVC, target_names=target_names))
rprt = classification_report(y_test, Y_predVC, target_names=target_names)
classi_rprt.plot_classification_report(rprt)

#Bar Graph
labels = ['DecisionTree','RandomForest','VotingClassifier']
sizes = [acc_dt,acc_rf,acc_vc]
index = np.arange(len(labels))
plt.bar(index, sizes)
plt.xlabel('Algorithm', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(index, labels, fontsize=10, rotation=0)
plt.title('comparative study')
plt.show()
