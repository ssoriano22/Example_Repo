#!/usr/bin/env python

#Keras Tutorial: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

print("Input Vars:",X)
print("Output Vars:",Y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINING: fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=10)

# VALIDATION: evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

# TEST/PREDICT: make probability predictions with the model
#predictions = model.predict(X)
# round predictions 
#rounded = [round(x[0]) for x in predictions]
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))