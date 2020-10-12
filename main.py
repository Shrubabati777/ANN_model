#importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv_def

#read the dataset and import
dataset = pd.read_csv('city.csv')
x=dataset.iloc[:,2:-1].values   #everything taken into account exept name of memory and remembers or not
y=dataset.iloc[:,-1].values     #only remembers or not column
# print(x)
# print(y)

#splitting into training and test set
sc=StandardScaler()
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#initialize the ANN
ann= tf.keras.models.Sequential()

#adding the input layer and one hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#adding the next hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) 

#adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling and resetting weights after error check (backward propagation)
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

#printing the final result
print(ann.predict(sc.transform([[0, 1, 0]])))
