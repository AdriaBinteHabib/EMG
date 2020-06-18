import pandas as pd
import numpy as np
import os

dataframes = []

for i in range(1, 37):
    path = str(i)+"/" # 1/
   # print(path)
    for file in os.listdir(path) :
       # print(file)
        data = pd.read_csv(path+file,sep='\s+') #1_raw_data_12-10_26.04.16.txt
       # data.columns = ['Time','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Channel7','Channel8','Class']
        dataframe = pd.DataFrame(data)
        dataframes.append(dataframe)
    # os.close(0)
        
result = pd.concat(dataframes)

result.replace([np.inf, -np.inf], np.nan)
result.dropna(inplace=True)
#Check unique values
#print(result.time.nunique())

#Creating Matrix of Features
x = result.iloc[:, :-1 ].values

'''#Standardization; working as feature scaling #NOT APPLICABLE, GIVES SAME RESULT .
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)'''

y = result.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

'''#Logistic Regression Algorithm
#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#predicting the test set results
y_pred= classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)'''

# K-Nearest Neighbors (K-NN) Algorithm
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)

"""#kernel SVM
#Fitting SVM to the training set
from sklearn.svm import SVC  
classifier= SVC(kernel= 'linear', random_state=0)
#x_train_shape= x_train.reshape(-1,1)
#y_train_shape= y_train.reshape(-1,1)
#print(x_train_shape)
#x_train_new= x_train_plus()
#y_train_new= y_train_plus()
classifier.fit(x_train, y_train)

#predicting the test set results
#x_test_shape= x_test.reshape(-1,1)
#x_test_new= x_test_plus()
y_pred= classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
#y_test_shape= y_test.reshape(-1,1)
#y_test_new= y_test_plus()
cm= confusion_matrix(y_test, y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)"""

"""#Naive Bayes Algorithm
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Predicting the Test set results
y_pred = classifier.predict(x_test.reshape(-1, 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.reshape(-1, 1), y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)"""

"""#Decision Tree Algorithm
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Predicting the Test set results
y_pred = classifier.predict(x_test.reshape(-1, 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.reshape(-1, 1), y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)"""

"""#Random Forest Algorithm
#Fitting RandomForest to the training set
from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators=10, criterion= 'entropy', random_state=0)
classifier.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

#predicting the test set results
y_pred= classifier.predict(x_test.reshape(-1, 1))

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test.reshape(-1, 1), y_pred)

#Calculating Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(accuracy)"""