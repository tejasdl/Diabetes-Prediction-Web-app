# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:58:53 2022

@author: Tejas Ligade
"""

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
#Loading the saved model 
loaded_mod = pickle.load(open("C:/Users/Tejas Ligade/diab_pred/trained_mod1.sav", 'rb')) #rb = reading binary

input_data = (4,110,92,0,0,37.6,0.191,30)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_mod.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')