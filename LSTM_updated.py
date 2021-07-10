#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
y_train = np.load('Data/y_train.npy')
y_test = np.load('Data/y_test.npy')


from keras.utils.np_utils import to_categorical
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:8]) #lable of first 8 row



from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization
from tensorflow.keras import regularizers
import tensorflow as tf 
import keras.callbacks as cb

# In[29]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[34]:
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


verbose, epochs, batch_size, rate, drop = 1, 1000, 512, .5, 0.1
hidden_activation = 'relu' 
out_activation = 'softmax'
loss_function = 'categorical_crossentropy'



# define a function to build the keras model
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
n_timesteps = x_train.shape[0]
n_features = x_train.shape[2]
n_outputs = Y_train.shape[0] 

print("time",n_timesteps)
print("features",n_features)
print("output",n_outputs)
model = Sequential()
model.add(LSTM(128, input_shape=(n_timesteps,n_features),return_sequences=False))
model.add(Dropout(drop))
model.add(Dense(256, activation=hidden_activation))
model.add(Dropout(drop))
model.add(Dense(512, activation=hidden_activation))
model.add(Dropout(drop))
model.add(Dense(1024, activation=hidden_activation))
model.add(Dropout(drop))
model.add(Dense(256, activation=hidden_activation))
model.add(Dropout(drop))
model.add(Dense(64, activation=hidden_activation))
model.add(Dropout(drop))
model.add(Dense(7, activation=out_activation))
opt = keras.optimizers.Adam(learning_rate=rate)
model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
# print(model.summary())
print('Verbose:',verbose)
print('Epoch:',epochs)
print('Batch Size:',batch_size)
print('Learning Rate:',rate)
print('Optimizer:','adam')
print('Drop Rate:',drop)
print('Activation (Hidden):',hidden_activation)
print('Activation (OutLayer):',out_activation)
print('Loss Function:',loss_function)

history = model.fit(x_train, Y_train, validation_data=(x_test, Y_test), epochs=epochs,batch_size=batch_size,callbacks=callbacks)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
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








