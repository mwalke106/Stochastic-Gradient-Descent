# Homework 1 Problem 4C

import numpy as np
import sklearn
import csv
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#sgd learning rate:
n = math.exp(-15)

#import data from csv file
with open('sgd_data.csv','r') as dest_f: 
    data_iter = csv.reader(dest_f)      
    data = [data for data in data_iter] 
data_array = np.asarray(data, dtype = float)

#start out w with these values
w = np.array([0.001, 0.001, 0.001, 0.001, 0.001])

#calculate the initial error
copy_data_init = data_array
np.random.shuffle(copy_data_init)
error_arr_init = []
for i in range(0, len(copy_data_init)):
    error_arr_init.append(np.square(np.subtract(copy_data_init[i, 5], np.dot(w, copy_data_init[i, 0:5]))))
prev_error = np.sum(error_arr_init)
current_error = prev_error

#initialize variables
round_ctr = 0   #epoch number



while True:
    error_arr = []
    copy_data = data_array
    prev_error = current_error
    #shuffle the data for stochastic gd
    np.random.shuffle(copy_data)
    #update the learning rate
    for i in range(0, len(copy_data)):
        w = w - n*np.dot(-2*(copy_data[i, 5] - np.dot(w, copy_data[i, 0:5])), copy_data[i, 0:5])
    #calculate the error for this epoch    
    for i in range(0, len(copy_data)):
        error_arr.append(np.square(np.subtract(copy_data[i, 5], np.dot(w, copy_data[i, 0:5]))))
    current_error = np.sum(error_arr)
    round_ctr += 1
    #if finished first epoch, calc delta_01 to find stop condition
    if round_ctr == 1:
        delta_01 = prev_error - current_error
    #stop condition - if error is small enough then break    
    if (prev_error - current_error)/delta_01 < 0.0001:
        break
        
       


