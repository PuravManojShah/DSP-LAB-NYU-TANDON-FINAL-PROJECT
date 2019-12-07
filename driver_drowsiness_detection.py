
# coding: utf-8

# In[1]:


import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import scipy
import playsound
# In[2]:

#pathname='/Users/puravshah/Downloads/decay_cosine_mono.wav'
#pathname='/Users/puravshah/Downloads/alarm_music.mp3'
pathname='/Users/puravshah/Downloads/Siren-SoundBible.com-1094437108.wav'
#alarm_on=False
def sound_the_alarm(path):
    playsound.playsound(path,block=False)
def predictortoxy(shape):
    arr=np.zeros((shape.num_parts,2),dtype='int')
    #arr=[]
    for i in range(0,shape.num_parts):
        arr[i]=(shape.part(i).x,shape.part(i).y)
    return arr
def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    return (A+B)/(2*C)
def boundingbox(detected):
    x=detected.left()
    y=detected.top()
    w=detected.right()-x
    h=detected.bottom()-y
    return (x,y,w,h)
aspect_ratio_threshold=0.20
no_of_frames=10
count=0
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#print(predictor)
cap=cv2.VideoCapture(0)
combined=0
while(True):
    _,frame=cap.read()
    #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(frame,0)
    for rect in rects:
    #Return the facial landmarks features using the predictor function,returns a shape object.
        shape=predictor(frame,rect)
        #print(shape.part(1).x)
    #Convert the returned facial landmarks to obtain (x,y) co-ordinates of the detected facial landmarks
        shape=predictortoxy(shape)
        (x,y,w,h)=boundingbox(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #(shape[36][0])
    #Get the left and the right eye co-ordinates
        left_eye_coord=shape[36:42]
        right_eye_coord=shape[42:48]
    #Calculate the left and the right eye aspect ratio
        aspect_ratio_left=eye_aspect_ratio(left_eye_coord)
        aspect_ratio_right=eye_aspect_ratio(right_eye_coord)
    #Calculate the average aspect ratio
        combined=(aspect_ratio_left+aspect_ratio_right)/2
        lefteyehull=cv2.convexHull(left_eye_coord)
        righteyehull=cv2.convexHull(right_eye_coord)
        #print('hull={}'.format(lefteyehull))
        #print('eye={}'.format(lefteye))
        #print(aspect_ratio_left,aspect_ratio_right)
        #righteye=cv2.convexHull(righteye)
        cv2.drawContours(frame,[lefteyehull],-1,(0,255,0),1)
        cv2.drawContours(frame,[righteyehull],-1,(0,255,0),1)
        #print(counter)
        if combined<aspect_ratio_threshold:
            count+=1
            #print(counter)
            if count>=no_of_frames:
                '''if pathname!='':
                    t=Thread(target=sound_the_alarm,args=(pathname,))
                    t.deamon=True
                    t.start()'''
                #alarm_on=True
                #if alarm_on:
                sound_the_alarm(pathname)
                cv2.putText(frame,'STOP SLEEPING!!WAKE UP!!',(30,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    #cv2.putText(frame,'WAKE THE F*** UP!!',(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            count=0
            #alarm_on=False
    #print(count)
    cv2.putText(frame,'aspect_ratio={}'.format(combined),(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow('stream',frame)
    key=cv2.waitKey(1)
    #print(key)
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

#/Users/puravshah/Downloads/decay_cosine_mono.wav
