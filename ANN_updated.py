#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:52:29 2021

@author: arifshakil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
y_train = np.load('Data/y_train.npy')
y_test = np.load('Data/y_test.npy')




from sklearn.model_selection import learning_curve

#from mlxtend.plotting import plot_learning_curves
import datetime



from keras.utils.np_utils import to_categorical
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:8]) #lable of first 8 row


# In[87]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf 


# In[88]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[108]:


# define a function to build the keras model
batch = 512
e=1000


rate = 0.001
drop_rate = 0.0001
# create model
model = Sequential()
model.add(Dense(4096, input_dim=9, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(2048, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(1024, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(256, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(128, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(64, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(128, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(256, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(1024, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(2048, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(4096, kernel_initializer='glorot_uniform', kernel_regularizer=None, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(7, activation='softmax'))

#compile model
opt = keras.optimizers.Adam(learning_rate=rate)
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



import keras.callbacks as cb
callbacks = [
    cb.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    )
]



history=model.fit(x_train, Y_train, validation_data=(x_test, Y_test),epochs=e, batch_size=batch, callbacks=callbacks)



# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()







# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(x_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))





