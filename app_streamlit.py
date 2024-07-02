import os
import streamlit as st
import numpy as np
from PIL import Image
from util import classify, set_background
import tensorflow as tf

set_background('app/bgs/eye2.png')
# print(os.getcwd())
# set title
st.title('Diabetic Retinopathy classification')
# set header
st.header('Please upload a Retinal Scan Image')
# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


# URL del modelo en Google Drive
model_url = 'https://drive.google.com/file/d/1U96luzv8S4RLlUI6np_ZR7JgmM8QduA3/view?usp=sharing'
