#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:57:34 2021

@author: arifshakil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
y_train = np.load('Data/y_train.npy')
y_test = np.load('Data/y_test.npy')


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import math 

#Applying LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda= LDA(n_components=6)
# x_train= lda.fit_transform(x_train, y_train)
# x_test= lda.transform(x_test)

# print("Applying PCA:", datetime.datetime.now())
# #Applying PCA
# from sklearn.decomposition import PCA
# pca= PCA(n_components=7)
# x_train= pca.fit_transform(x_train)
# x_test= pca.transform(x_test)
# explained_variance= pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0, C=3, max_iter= 500)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)












