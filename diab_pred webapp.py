# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:05:37 2022

@author: Tejas Ligade
"""

import numpy as np
import pickle 
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()


#Loading the saved model 
loaded_mod = pickle.load(open("C:/Users/Tejas Ligade/diab_pred/trained_mod1.sav", 'rb')) #rb = reading binary

#Creating a function for prediction 

def diab_pred(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_mod.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'
      
def main():
    
    
    #Giving a title
    st.title('Diabetes Prediction Web App')
    
    #Getting input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    #Code for prediction
    diagnosis = ''
    
    #Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diab_pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)    
    


if __name__ == '__main__':
    main()
    

     
      
      
      
      
      