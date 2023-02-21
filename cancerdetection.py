# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:42:08 2023

@author: mayank
"""

import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array
from keras.models import load_model

import numpy as np
import os
from PIL import Image


loaded_model = pickle.load(open('trained_model.sav', 'rb'))
def cancer_detection(input_data):
    
    input_data_numpy_array = np.asarray(input_data)
    input_data_reshape = input_data_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)

    if (prediction[0]== 0):
        return "The person doesnot have Lung Cancer"
    else:
        return "The person may be suffering from Lung Cancer"
        
def cancer_image(img_file_buffer): 
    resnet50_model = load_model("finalmodel-ResNet50.hdf5")
    img = Image.open(img_file_buffer)
    img = img.convert('RGB')
    img = img.resize((460,460))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    class_names=['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma'] 
    prediction = resnet50_model.predict(img_array)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(prediction)], 100 * np.max(prediction))
    )

def main():
    st.title('cancer prediction app')
    gender=st.selectbox('0 for men 1 for female',[0,1])
    age=st.number_input('enter age',0,100)
    smoking=st.selectbox('if smoking (1 for no,2 for yes)',[1,2])
    yellow_fingers=st.selectbox('if yellow_fingers (1 for no,2 for yes)',[1,2])
    anxiety=st.selectbox('if anxiety (1 for no,2 for yes)',[1,2])
    peer_pressure=st.selectbox('if peer_pressure (1 for no,2 for yes)',[1,2])
    chronic_disease=st.selectbox('if chronic_disease (1 for no,2 for yes)',[1,2])
    fatigue=st.selectbox('if fatigue (1 for no,2 for yes)',[1,2])
    allergy=st.selectbox('if allergy (1 for no,2 for yes)',[1,2])
    wheezing=st.selectbox('if wheezing (1 for no,2 for yes)',[1,2])
    alcohol=st.selectbox('if alcohol (1 for no,2 for yes)',[1,2])
    coughing=st.selectbox('if coughing (1 for no,2 for yes)',[1,2])
    shortnessofbreath=st.selectbox('if shortnessofbreath (1 for no,2 for yes)',[1,2])
    swallowing_difficulty=st.selectbox('if swallowing_difficulty (1 for no,2 for yes)',[1,2])
    chestpain=st.selectbox('if chestpain (1 for no,2 for yes)',[1,2])
    inputimage = st.file_uploader('Upload chest CT')
    
    
    diagnosis=''
    
    if st.button('cancer test result'):
        diagnosis=cancer_detection([gender,age,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatigue,allergy,wheezing,alcohol,coughing,shortnessofbreath,swallowing_difficulty,chestpain])
        image_diagnosis = cancer_image(inputimage)

    
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    