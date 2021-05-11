# face_detector
This is a repository that I created on this day of 9th feb


import cv2
import random

# load some pre-trained data on face formate from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#  choose an image to decect faces in
# img = cv2.imread('rdj.png')
# img = cv2.imread('woman.png')
# img = cv2.imread('baby.png')
# img = cv2.imread('human.png')
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('clipi.mp4')

while True:

    # read the current frame
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangle around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x, y) , (x+w, y+h), (random.randrange(256),random.randrange(256),random.randrange(256)), 2) 

    # for show phot
    cv2.imshow('Clever Programmer Face Detecter', frame)
    cv2.waitKey(1)
    
    # # stop if 0 key is pressed
    # if key==81 or key==113:
    #     break


webcam.release()

   

"""
# must convert to bgrayCale
grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Detect the face
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
print(face_coordinates)
(x, y, w, h) = face_coordinates[1]
cv2.rectangle(img,(x, y) , (x+w, y+h), (0,0,255), 2)
# cv2.rectangle(img,(206 , 18) , (206+43, 18+43), (0,0,255), 2)
# for show phot
cv2.imshow('Clever Programmer Face Detecter', img)
# 
cv2.waitKey()
print("code completes") 
"""
