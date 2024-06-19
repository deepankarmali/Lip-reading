import streamlit as st 
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model 

#Wide Layout 
st.set_page_config(layout = 'wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Buddy')
    st.info("This application is originally developed from the LipNet Deep learning model.")

st.title("LipNet Full Stack App")

#Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_videos = st.selectbox('Choose Video', options)

#Generate 2 columns
col1, col2 = st.columns(2)

if options:
    #Rendering the Video
    with col1:
        st.info('The Video below displays the video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_videos)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        #Rendering inside streamlit 
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
        
    with col2:
        st.info("This is all the machine learning model sees when making a prediction")
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', np.squeeze(video), fps = 10)
        st.image('animation.gif', width = 400)
        st.info("This is the output of the machine leanring model as tokens")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy = True)[0][0].numpy()
        st.text(decoder)

        #Converted Prediction
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join((num_to_char(decoder))).numpy().decode('utf-8')
        st.text(converted_prediction)
        