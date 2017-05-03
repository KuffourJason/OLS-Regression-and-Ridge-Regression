from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loads the dataset
data = np.loadtxt("airfoil_self_noise.dat")

#separates the features and output values
inp = data[:,0:5]
out = data[:,5]

#Splits dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(inp, out,  random_state=0)

print(inp.shape)

#Trains the model with the training set 
#Uses ridge regression which is MLE 
lr = Ridge().fit(x_train, y_train)

#w parameter is stored in lr.coef_
#b parameter is stored in lr.intercept_

print("Training set score: {:.2f}".format(lr.score(x_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(x_test, y_test)))