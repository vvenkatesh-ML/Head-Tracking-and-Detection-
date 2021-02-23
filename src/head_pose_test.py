# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:59:50 2021

@author: khasy
"""

import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from scipy.spatial import distance as dist
import dlib
from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import img_to_array
from keras.models import load_model

detector = MTCNN() #Very good yaw and pitch angles
detector_2 = dlib.get_frontal_face_detector() #Blink/sleep detection
predictor_2 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

font = cv2.FONT_HERSHEY_COMPLEX
font_size= 0.4
blue= (225,0,0)
red = (0,0,255)
yellow = (0,255,255)
orange = (0,155,255)
green=(0,128,0)
green_2 = (0, 255, 0)

#Eye Parameters
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
EYE_AR_RATIO = 0.22
EYE_AR_CONSEC_FRAMES = 1
SLEEP_FRAMES = 5

#Eye counter Variables
counter = 0
total = 0

#Calculates eye aspect ratios
def eye_aspect_ratio(frame, eye):
    #Compute euclidean distance between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    #Compute euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B)/(2 * C)
    return ear    

#Draw eye contours
def draw_eye_contours(frame, rects):
    for rect in rects:
        #get facial landmarks
        landmarks = np.matrix([[p.x, p.y] for p in predictor_2(frame, rect).parts()])
        #get left eye landmarks
        left_eye = landmarks[LEFT_EYE_POINTS]
        #get right eye landmarks
        right_eye = landmarks[RIGHT_EYE_POINTS]
        #draw contours on the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, green_2, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, green_2, 1)    
        return left_eye, right_eye   
    

#Detect face and prepare data 
def face_points(frame,bbs,points):
    bb = bbs[0]
    keypoints = points[:,0]
    return bb, keypoints

#Draw rectangle and facial landmarks
def draw(frame,bb,keypoints):
    
    cv2.rectangle(frame, (int(bb[0]),int(bb[1])), (int(bb[2]), int(bb[3])), orange, 2)
                 
    cv2.circle(frame, (keypoints[2], keypoints[7]),2,yellow,2) #Nose
    cv2.circle(frame, (keypoints[3], keypoints[8]),2,yellow,2) #Left Mouth
    cv2.circle(frame, (keypoints[4], keypoints[9]),2,yellow,2) #Right Mouth 
    
def apply_offsets(frame, bb):
    x1 = int(bb[0])
    y1 = int(bb[1])
    x2 = int(bb[2])
    y2 = int(bb[3])
    return x1,x2,y1,y2
    
#Change of lengths between each eye wrt nose
def yaw(pts):
    l_eye_nose = pts[2] - pts[0]
    r_eye_nose = pts[1] - pts[2]
    diff_eyes_nose = l_eye_nose - r_eye_nose
    return diff_eyes_nose

#Change of lengths between eyes and mouth wrt nose
def pitch(pts):
    avg_eye_y = (pts[5] + pts[6])/2
    avg_mouth_y = (pts[8] + pts[9])/2
    e2n = avg_eye_y - pts[7]
    n2m = pts[7] - avg_mouth_y
    y_diff = e2n - n2m
    return y_diff

#Creates a list of the desired variables
def dataframe(yaw, pitch, label, label_2, timestamp):
    yaw_angle.append(yaw)
    pitch_angle.append(pitch)      
    yaw_label.append(label)
    pitch_label.append(label_2)
        
    Time_stamp.append(timestamp)
    return Time_stamp, yaw_angle, pitch_angle, yaw_label, pitch_label


video_save = True

fps = 10
video_format = cv2.VideoWriter_fourcc('M','J','P','G')
video_max_frame = 10
video_outs=[]
              
emotion_model_cnn = 'models/mini_xception.h5'    
emotion_classifier = load_model(emotion_model_cnn, compile=False)

emotions = ['Angry', 'Disgust', 'Scared', 'Happy', 'Neutral', 'Surprised', 'Sad']


#Start Video Camera Feed

cap = cv2.VideoCapture(0)    

if video_save:
    video_file='video_out.avi'
    video_out = cv2.VideoWriter(video_file, video_format, fps, (640,480))

yaw_angle = []
pitch_angle = []
Time_stamp = []
yaw_label =[]
pitch_label = []

 
while True:
    #Capture Face
    ret,frames = cap.read()
    frame = np.array(frames)
    frame = cv2.flip(frame,1)
    timestamp = datetime.now()
    
    #Grayscale conversion
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector_2(gray_image, 0)

    #Use MTCNN to detect faces
    bbs, points = detector.detect_faces(frame)
    


    if len(bbs) > 0:
        bb, keypoints = face_points(frame,bbs,points)
        
        #Emtotion Detection 
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags = cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            im = gray[y:y + h, x:x + w]
            im = cv2.resize(im, (48,48))
            im = im.astype("float")/255.0
            im = img_to_array(im)
            im = np.expand_dims(im, axis=0)
            preds = emotion_classifier.predict(im)[0]
            emotion_probability = np.max(preds)
            em_label = emotions[preds.argmax()]
        
       
        #Eye Blink Values
        try:
            left_eye, right_eye = draw_eye_contours(frame, rects) 
                     
        # Compute EAR
            ear_left = eye_aspect_ratio(frame, left_eye)
            ear_right = eye_aspect_ratio(frame, right_eye)
            #average EAR
            ear_avg = round((ear_left + ear_right)/2.0,2)
                
            #Detect eye blink & sleep 
            if ear_avg < EYE_AR_RATIO:
                counter += 1
            else:
                if counter >= EYE_AR_CONSEC_FRAMES:
                    total += 1
                if counter >= SLEEP_FRAMES:
                    sleep_total = 'Yes'
                else:
                    sleep_total = 'No'
                counter = 0  
        except:
            pass
            
        #Draw keypoints and bounding box
        draw(frame,bb,keypoints)
   
        #Head Rotation Labels
        if yaw(keypoints) > 10:
            label = 'Head Left'
        elif yaw(keypoints) < -10:
            label = 'Head Right'
        else:
            label = 'No Major Movement'
    
        if pitch(keypoints) > 5:
            label_2 = 'Head up'
        elif pitch(keypoints) < -5:
            label_2 = 'Head Down'
        else:
            label_2 = 'No Major Movement '
   
        cv2.putText(frame, "Yaw: {0:.2f}".format(yaw(keypoints)), (10,100), font, font_size, blue, 1)
        cv2.putText(frame, "Pitch: {0:.2f}".format(pitch(keypoints)), (10,120), font, font_size, blue, 1)        
        cv2.putText(frame, 'Yaw Label: '+label, (10,140), font, font_size, red, 1)
        cv2.putText(frame, 'Pitch Label: '+label_2, (10,160), font, font_size, red, 1)
        cv2.putText(frame, 'Time: '+str(timestamp), (200,10), font, font_size, green, 1)
        cv2.putText(frame, 'Emotion: '+em_label, (10,180), font, font_size, orange, 1)
        cv2.putText(frame, 'Blink: {} '.format(total), (10,200), font, font_size, blue, 1)
        cv2.putText(frame, 'Eye AR: {} '.format(ear_avg), (10,220), font, font_size, blue, 1)
        cv2.putText(frame, 'Sleep: '+sleep_total, (10,240), font, font_size, blue, 1)     

    cv2.imshow('Face_Detector',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

    if video_save:
        video_out.write(frame)

    Time_stamp, yaw_angle, pitch_angle, yaw_label, pitch_label = dataframe(yaw(keypoints),pitch(keypoints),label,label_2, timestamp)

df = pd.DataFrame({'Timestamp':Time_stamp, 'Yaw Angle': yaw_angle, 'Pitch Angle': pitch_angle, 'Yaw Label': yaw_label, 'Pitch Label': pitch_label})
df.to_csv(r'C:\Users\khasy\OneDrive\Documents\Head Tracking System\head_pose_vishaal\angles_out.csv', index = False, header = True)
    

cap.release()

if video_save:
    video_out.release()

cv2.destroyAllWindows()
            
                     