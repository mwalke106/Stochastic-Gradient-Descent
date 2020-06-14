 # Homework 1 Problem 4D
 
import numpy as np
import sklearn
import csv
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 


#import data from csv file 
with open('sgd_data.csv','r') as dest_f: 
    data_iter = csv.reader(dest_f)      
    data = [data for data in data_iter] 
data_array = np.asarray(data, dtype = float)


#initialize variables
round_ctr = 0   #epoch number
#array of different learning rates to be used
eta = [math.exp(-10), math.exp(-11), math.exp(-12), math.exp(-13), math.exp(-14), math.exp(-15)]
#these arrays will hold the squared loss for each learning rate
error10 = []
error11 = []
error12 = []
error13 = []
error14 = []
error15 = []

 
for b in range(0, len(eta)):
    round_ctr = 0  
    w = np.array([0.001, 0.001, 0.001, 0.001, 0.001]) #reset the w array to init
    while round_ctr < 500:   #500 was chosen rand, for e^-15 num epochs was 766
        error_arr = []
        copy_data = data_array
        np.random.shuffle(copy_data)
        #calculate the error before the epoch
        for i in range(0, len(copy_data)):
            error_arr.append(np.square(copy_data[i, 5] - np.dot(w, copy_data[i, 0:5])))
        #place error in right array depending on learning rate used
        if b == 0:
            error10.append(np.sum(error_arr))
        elif b == 1:
            error11.append(np.sum(error_arr))
        elif b == 2:
            error12.append(np.sum(error_arr))
        elif b == 3:
            error13.append(np.sum(error_arr))
        elif b == 4:
            error14.append(np.sum(error_arr))
        elif b == 5:
            error15.append(np.sum(error_arr))
        #update weight vector using gradient descent
        for i in range(0, len(copy_data)):
            w = w - eta[b]*np.dot(-2*(copy_data[i, 5] - np.dot(w, copy_data[i, 0:5])), copy_data[i, 0:5])
        round_ctr += 1
     
epochs = list(range(500))

#plot all errors w/ corresp. learning rate on one plot
plt.plot(epochs, error10, 'r--', epochs, error11, 'bs', epochs, error12, 'g^', epochs, error13, 'cs', epochs, error14, 'k-', epochs, error15, 'm^')
plt.show()