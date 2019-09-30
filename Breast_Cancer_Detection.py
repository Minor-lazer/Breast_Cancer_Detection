import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing,cross_validation
from sklearn.neighbours import KNeighboursClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli','mitoses', 'class']
dataset=pd.read_csv(url,names=names)
dataset.replace('?',-99999,inplace=True)
print(dataset.axes)
dataset.drop(['id'],1,inplace=True)
print(dataset.loc[9])
print(dataset.shape)
print(dataset.describe())
dataset.hist(figsize=(10,10))
plt.show()

scatter_matrix(dataset,figsize=(18,18))
plt.show()
X=np.array(dataset.drop(['class'],1))
Y=np.array(dataset['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

seed=8
scoring='accuracy'

models=[]
models.append(('KNN',KNeighboursClassifier(n_neighbours=5)))
models.append(('SVM'),SVC())

result=[]
names=[]

for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
    