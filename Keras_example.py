#!/usr/bin/env python

#Keras Tutorial: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# first neural network with keras tutorial
from numpy import loadtxt
import numpy
import os
from tensorflow.keras.models import Sequential, model_from_json, save_model, load_model
from tensorflow.keras.layers import Dense

# fix random seed for reproducibility
numpy.random.seed(7)

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

# MODEL SAVE 1: separate model (JSON) + weights (HDF5) files
# Save model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Save weights to HDF5
model.save_weights("model.h5")
print("Saved model files to disk")

# MODEL SAVE 2: single trained keras model file
#model.save("model_complete.h5")
# equivalent to this keras function:
save_model(model, "model_complete.h5")
print("Saved complete model to disk")

# MODEL SAVE 3: Protocol Buffer format - used by some TensorFlow 
#               pre-trained models available for download
# save model and architecture W/O .hd5 ending
# will result in "directory" of files for model
model.save("model_PBformat")

#......Later.............
print("Some time later.........")

# MODEL LOAD 1: separate model + weights files
print("MODEL LOAD 1 - Multiple Files (JSON/HDF5):")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model files from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# MODEL LOAD 2: single keras model file
print("MODEL LOAD 2 - Single File Keras:")
# load model
loaded_model2 = load_model("model_complete.h5")
# summarize model.
loaded_model2.summary()
# load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# evaluate the model
score = loaded_model2.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model2.metrics_names[1], score[1]*100))

# MODEL LOAD 3: PB format - "directory" model files (i.e. TensorFlow pre-trained models)
print("MODEL LOAD 3 - PB format:")
# load model
model_PB = load_model("model_PBformat")
# print summary
model_PB.summary()