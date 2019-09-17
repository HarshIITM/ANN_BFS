# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:53:49 2019

****'Artificial neural network for backward facing step'****

@author: Harsh Arora
"""

#data loading and preprocessing
import pandas as pd 
import glob
import os

datadir=(r'C:\Users\Harsh Arora\Desktop\ML-python\PINN\simulation_data')
filenames=glob.glob1(datadir,'*.csv')
dataset=pd.read_csv(r'C:\Users\Harsh Arora\Desktop\ML-python\PINN\simulation_data\t_5.csv',header=None, index_col=None)
for f in range(1,15):
    f1=os.path.join(datadir,filenames[f])
    f2=pd.read_csv(f1,header=None, index_col=None)
    dataset=pd.concat([dataset,f2],axis=0, sort=False)

# Defining input and output features        
X=dataset.iloc[:,0:3].values 
Y=dataset.iloc[:,3:7].values

# splitting into training set and testing(validation) set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling- coz we don't want one feature to dominate the other 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Y_train = sc.fit_transform(Y_train)
Y_test = sc.transform(Y_test)

# Defining the ANN architecture
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

main_input = Input(shape=(3,))
hidden1 = Dense(256, activation='tanh',kernel_initializer="normal")(main_input)
hidden2 = Dense(256, activation='tanh')(hidden1)
main_output = Dense(4, activation='sigmoid')(hidden2)

#compiling the model
model = Model(inputs=main_input, outputs=main_output)
print(model.summary())

#defining a custom loss function
import keras.backend as K
def custom_loss(y_true, y_pred):
    K.print_tensor(y_pred)
    return K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))

#compiling the model
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
#fitting the data
model.fit(X_train, Y_train, batch_size = 12, epochs = 100, validation_data=(X_test,Y_test))

# saving model(weights)
model_json = model.to_json()
with open("ANN_bfs_model.json", "w") as json_file:
    json_file.write(model_json)

#serialize weights to HDF5
model.save_weights("ANN_bfs_model.h5",overwrite=True)
print("Saved model to disk")

# loading a saved model architecture (json)
from keras.models import model_from_json
json_file = open('ANN_bfs_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# loading weights into new model
loaded_model.load_weights("ANN_bfs_model.h5")
loaded_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print("Loaded and compiled model from disk")

#importing and rearranging data for prediction
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
import pandas as pd
X_predict=pd.read_csv(r'C:\Users\Harsh Arora\Desktop\ML-python\PINN\predict.csv',header=None, index_col=None)
X_predict=X_predict.iloc[:,0:3].values 
X_predict = sc.fit_transform(X_predict)

#fixing the min-max scaling issue manually
for i in range(1581):
    X_predict[i,2]=0.4922
    
#predicting using loaded model
Y_pred = loaded_model.predict(X_predict)

u_pred=Y_pred[:,0]              #predicted u velocity
v_pred=Y_pred[:,1]              #predicted v-velocity
stream_pred=Y_pred[:,2]         #predicted stream function
vort_pred=Y_pred[:,3]           #predicted vorticity function

#analyzing predicted stream function
s_min=-0.13763
s_max=1.0002954
import numpy as np
s_transformed=np.zeros((1581,))

#inverse transformation to get the original values
for i in range(1581):
    s_transformed[i] = stream_pred[i]*(s_max-s_min) + s_min

#reshaping the vector into grid
s_grid=np.reshape(s_transformed,(51,31))

for i in range(16):
    for j in range(16):
        s_grid[i,j]=0

#plotting stream lines
import matplotlib.pyplot as plt
x1=np.linspace(0,5.0,51)
y1=np.linspace(0,3.0,31)
yv,xv=np.meshgrid(y1,x1)
plt.contour(xv,yv,s_grid,50)
















