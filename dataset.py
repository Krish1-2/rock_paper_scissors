import os
import mediapipe as mp
import numpy as np
import cv2

mp_drawing =mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

vedio=cv2.VideoCapture(0)

count=0
directory = 'dataset'
path=directory+'/scissors'

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5,max_num_hands=1) as hands:
    while(True):
        res,frame=vedio.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
        cv2.waitKey(300)
        x1, y1 = 50, 50  
        x2, y2 = 350, 350  

        roi = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('rock',image)

        if not os.path.exists(path):
                os.makedirs(path)
        cv2.imwrite(path+"/image"+str(count)+'.jpg',roi)
        count+=1
        if count==300:
            break

vedio.release()
cv2.destroyAllWindows()