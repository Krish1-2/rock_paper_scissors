import cv2
import random
import mediapipe as mp
import time
import tensorflow as tf
import pickle 
import numpy as np
import os 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print("Before loading the model")
savedModel = tf.keras.models.load_model('game_model.h5')
print("After loading the model")
with open("label_encoder.pkl", 'rb') as file:
    label_encoder = pickle.load(file)

list1 = ['rock', 'paper', 'scissors']
random_sel = random.choice(list1)
predicted_class=''
vedio = cv2.VideoCapture(0)

directory='predict_images'

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while True:
        ret, frame = vedio.read()
        x1, y1 = 25, 25  
        x2, y2 = 325, 325  
        cv2.waitKey(1000)
        
        while True:
            ret,captured_frame = vedio.read()
            image = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            roi = image[y1:y2, x1:x2]
            text_to = "place your hand in the box in and then press q"
            cv2.putText(image, text_to, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Captured Image', image)
            cv2.imshow('roi',roi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(directory+"/image"+'.jpg',roi)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_normalized = img_gray / 255
                resized_img = cv2.resize(img_normalized, (256, 256))
                resized_img = np.expand_dims(resized_img, axis=-1)
                resized_img = np.expand_dims(resized_img, axis=0)
                predict = savedModel.predict(resized_img)
                predicted_class_index = np.argmax(predict[0])
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                print("Predicted Gesture:", predicted_class)
                break

        print("Random Gesture:", random_sel)
        if((predicted_class=='rock' and random_sel=='scissor') or (predicted_class=='paper' and random_sel=='rock') or (predicted_class=='scissor' and random_sel=='paper')):
            print('you win')
        elif((predicted_class=='rock' and random_sel=='rock') or (predicted_class=='paper' and random_sel=='paper') or (predicted_class=='scissors' and random_sel=='scissors')):
            print('its a draw!')
        else:
            print('you lose')
        cv2.waitKey(2000)
        break

vedio.release()
cv2.destroyAllWindows()
