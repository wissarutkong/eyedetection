import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import playsound
from threading import Thread
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import socket
import pickle
import struct 

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close','Open']
model = load_model('models/WisCnn.h5')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
eye_lose=0
rpred=[99]
lpred=[99]

HOST=''
PORT=8485

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('[INFO] Socket created')

s.bind((HOST,PORT))
print('[INFO] Socket bind complete')
s.listen(10)
print('[INFO] Socket now listening')

conn,addr=s.accept()
print('[INFO] Ip address connect :', addr)

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

while(True):
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    frame = imutils.resize(frame, width=450)
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    if(int(len(left_eye)) == 0 and int(len(right_eye)) == 0 ):
        eye_lose=eye_lose+1
    else:
        eye_lose=0
        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(67,67))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(67,67,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(67,67))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(67,67,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            break
        
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
    else:
        score=0

    if(score>30 or eye_lose>30):
        try:
            conn.send("B".encode())
        except:
            pass

    data_recive = " "
    conn.send(data_recive.encode())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
conn.close();
cap.release()
cv2.destroyAllWindows()
