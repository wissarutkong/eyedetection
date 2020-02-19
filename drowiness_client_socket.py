import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from threading import Thread
from pygame import mixer

time.sleep(0.1)  

mixer.init()
sound = mixer.Sound('alarm.wav')

ALARM_ON = False
last_frame = None

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('YOUR_IP', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

def get_frame():
    global last_frame
    while True:
        ret, frame = cam.read()
        last_frame = frame

cam.set(3, 640);
cam.set(4, 540);

img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

get_frame_thread = Thread(target=get_frame)
get_frame_thread.start()


while True:
    if last_frame is not None:
        frame = last_frame
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame, 0)
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)
        datatext = client_socket.recv(4096)
        
        datatext = datatext.decode("utf-8").replace(' ','')
        if(datatext == 'B'):
            sound.play()

        img_counter += 1

cam.release()
