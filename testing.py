import cv2
import random
import mediapipe as mp
import time
import tensorflow as tf
import pickle 
import numpy as np
import os 
from keras.preprocessing.image import img_to_array

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("Before loading the model")
savedModel = tf.keras.models.load_model('game_model.h5')
print("After loading the model")
with open("label_encoder.pkl", 'rb') as file:
    label_encoder = pickle.load(file)

count=1
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while True:
        
        image=cv2.imread('predict_images/image'+str(count)+'.jpg')
        cv2.imshow('pic',image)
        cv2.waitKey(1000)
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray / 255
        resized_img = cv2.resize(img_normalized, (256, 256))
        resized_img = np.expand_dims(resized_img, axis=-1)
        resized_img = np.expand_dims(resized_img, axis=0)
        predict = savedModel.predict(resized_img)
        print(predict)
        predicted_class_index = np.argmax(predict[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        print("Predicted Gesture:", predicted_class)
        count+=1
        if count==9:
            break