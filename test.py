import face_recognition
import cv2, dlib
import numpy as np
import glob
import os
import logging
from imutils import face_utils
from keras.models import load_model
from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
import random
from random import randrange
import time

IMG_SIZE = (24, 24)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2019_08_01_12_11_29.h5')
model.summary()

IMAGES_PATH = './faces'  # put your reference images in here
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict
authNumber = 0

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
  return eye_img, eye_rect

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings
    

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}

    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]

        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[identity] = encodings[0]

    return database

def paint_detected_face_on_image(frame, location, name=None):
    global authNumber
    
    """
    Paint a rectangle around the face and write the name
    """
    
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    cv2.putText(frame, "please blink "+str(authNumber)+" to be authenticated", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    
def run_face_recognition_and_blink_detiction(database):
    """
    Start the face recognition via the webcam
    """
    
    print("here in threadVideoGet !!!!!!!!!!!!")
    video_getter = VideoGet(CAMERA_DEVICE_ID).start()
    cps = CountsPerSec().start()
    
    # Open a handler for the camera
    img_array = []
    TOTAL=0
    lastState=0
    COUNTER=0
    global authNumber
    
    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    #print(len(database.keys()))
    time.sleep(2)
    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())


        # run detection and embedding models
        face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

        # Loop through each face in this frame of video and see if there's a match
        for location, face_encoding in zip(face_locations, face_encodings):

            # get the distances from this encoding to those of all reference images
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # select the closest match (smallest distance) if it's below the threshold value
            if np.any(distances <= MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
                if(authNumber is 0):
                    authNumber=random.randint(1,3)
                print("found "+name)
                print("please blink "+str(authNumber)+" to be authenticated !!!!")
            else :
                authNumber=0
                name = None
                print(name)
                TOTAL=0
                COUNTER=0
                print("here")
            # put recognition info on the image
            paint_detected_face_on_image(frame, location, name)
        frame = cv2.resize(frame, dsize=(250, 250), fx=0.5, fy=0.5)
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        for face in faces:
          shapes = predictor(gray, face)
          shapes = face_utils.shape_to_np(shapes)
          # extract the left and right eye coordinates, then use the
          # coordinates to compute the eye aspect ratio for both eyes
          leftEye = shapes[36:42]
          rightEye = shapes[42:48] 
          eye_img_l, eye_rect_l = crop_eye(gray, leftEye)
          eye_img_r, eye_rect_r = crop_eye(gray, rightEye)

          eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
          eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
          eye_img_r = cv2.flip(eye_img_r, flipCode=1)
          cv2.imshow('l', eye_img_l)
          cv2.imshow('r', eye_img_r)
          eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
          eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

          pred_l = model.predict(eye_input_l)
          pred_r = model.predict(eye_input_r)

          state_l=False
          state_r=False
          if(pred_r<=0.05):state_r=False
          else:state_r=True
    
          if(pred_l<=0.05):state_l=False
          else:state_l=True
    
          if (COUNTER>=2 and lastState==1):
               TOTAL+=1
               COUNTER=0
          if(state_l==False and state_r==False):
               COUNTER+=1      
               lastState=0
          elif (state_l==True and state_r==True):
              lastState=1

          if (TOTAL>=authNumber and authNumber >0):
              print("DONE")
              cv2.putText(img, "AUTHENTICATED", (100,90), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 0), 1)
              TOTAL=0
          leftEyeHull = cv2.convexHull(leftEye)
          rightEyeHull = cv2.convexHull(rightEye)
           
          cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
          cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

          cv2.putText(img, str(state_l), tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
          cv2.putText(img, str(state_r), tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        height, width, layers = img.shape
        size = (width,height)
        cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        img_array.append(img)
        cv2.imshow('Result', img)
        cps.increment()
        
        
        out = cv2.VideoWriter('result_m5.avi',cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
          
       
  # When everything done, release the capture
    
    cv2.destroyAllWindows()
  
    print('a')            
        


############################################main###########################3
database = setup_database()
run_face_recognition_and_blink_detiction(database)
