
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from depthCameraClass import *
from HandTrackingClass import *
from PoseTrackingClass import *
from holisticTrackingClass import *

camera = DepthCamera()
handDetect=HandDetector(False,4,1,0.5,0.5)
poseDetect=PoseDetector(False,1,True,False,True,0.5,0.5)
holisticDetect =HolisticDetector(False,1,True,False,True,False,0.5,0.5)

controlActions = np.array(["Reverse","Forward","TurnLeft", "TurnRight","Stop","Takeover","Release"])
numVideos = 30 #number of videos fr each action
numFrames = 28 #number of frames in one video 

colors = [(245,117,16), (117,245,16), (16,117,245),(50,168,82),(82,50,168),(168,82,50),(25,200,112)]

def confi_visualization(predictions, ctrlActions, color_img, colors):
      output = color_img.copy()
      for num, prob in enumerate(predictions):
            cv2.rectangle( output,(0,60+num*40),(int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output,ctrlActions[num],(0,85+num*40),cv2.FONT_HERSHEY_SIMPLEX,1, (25,255,255), 2, cv2.LINE_AA)
      return output
    
             
      
   
frames = []
sentence = []
threshold = 0.7

l2=regularizers.L2(1e-5)
l1l2=regularizers.L1L2(l1=1e-4, l2=1e-5)

model = Sequential()
model.add(LSTM(128,return_sequences=True,kernel_regularizer= l2, activation='elu', input_shape=(28,126)))
#model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(40,1662)))
#model.add(LSTM(64,return_sequences=True,kernel_regularizer= l2,  activation='elu'))
model.add(LSTM(32,return_sequences=False,kernel_regularizer= l2, activation='elu'))
#model.add(Dense(64,kernel_regularizer= l2, activation='elu'))
model.add(Dense(32,kernel_regularizer= l2, activation='elu'))
model.add(Dense(controlActions.shape[0], activation='softmax'))

adam =Adam(learning_rate=0.00001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('GestureRecognition_2layers.h5')

cap  = cv2.VideoCapture('IMG_1191.mp4')
while cap.isOpened():
    ret,color_img = cap.read()
    #ret, depth_img, color_img = camera.get_frame()

    #hand_img = handDetect.find_hand(color_img)
    #cv2.imshow("hand detected image", hand_img)

    #pose_img = poseDetect.find_pose(color_img)
    #cv2.imshow("pose detected image", pose_img)
    holistic_img = holisticDetect.find_holistic(color_img)
    pose,leftHand, rightHand, face = holisticDetect.find_location()
    #converting the list to numpy array and flattened for one dimension
    pose = np.array(pose).flatten()
    leftHand = np.array(leftHand).flatten()
    rightHand = np.array(rightHand).flatten()
    face = np.array(face).flatten()
    #allKeypoints = np.concatenate([pose,leftHand,rightHand,face])
    allKeypoints = np.concatenate([leftHand,rightHand])

    frames.append(allKeypoints)
    print(len(frames))
    #frames.insert(0,allKeypoints)
    #frames.insert(len(frames),allKeypoints)
    #frames.append(allKeypoints)
    #frames = frames[:numFrames]
    
    if len(frames) >= numFrames:
          
          prediction = model.predict(np.expand_dims(frames[-28:], axis=0))[0]

          if prediction[np.argmax(prediction)] > threshold :
               print(controlActions[np.argmax(prediction)])

          holistic_img = confi_visualization(prediction, controlActions, holistic_img, colors)
          #frames.clear()

    '''
   
    if len(frames) == numFrames:
          prediction = model.predict(np.expand_dims(frames, axis=0))[0]
          print(controlActions[np.argmax(prediction)])

          if prediction[np.argmax(prediction)] > threshold :
                  if len(sentence) > 0:
                        if controlActions[np.argmax(prediction)] != sentence[-1]:
                              sentence.append(controlActions[np.argmax(prediction)])
                  else:
                        sentence.append(controlActions[np.argmax(prediction)])

          if len(sentence) > 5 : 
                  sentence = sentence[-5:]

          holistic_img = confi_visualization(prediction, controlActions, holistic_img, colors)
          frames.clear()
'''
    cv2.imshow("pose detected image", holistic_img)

    #cv2.imshow("color image", color_img)
    #cv2.imshow("depth image", depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


