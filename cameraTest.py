"""
////////////////////////////////////////////////////////////
||| Date: 03/09/2023                                    ///
||| Author: Herve Mwunguzi                             ///
||| Purpose: Testing the Intel realsense depth camera ///
|||/////////////////////////////////////////////////////

"""

import pyrealsense2.pyrealsense2 as rs
import time
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

def drawBoundingBox(img,landmarklist):
      h,w,c = img.shape
      try:  
        return cv2.rectangle(img,(int(landmarklist[4][0]*w),int(landmarklist[12][1]*h)),(int(landmarklist[20][0]*w),int(landmarklist[0][1]*h)), (255,0,0),2)
      except IndexError:
        return img 

# cap = cv2.VideoCapture("./TrainingVideos/IMG_1187.mp4")


while True:
    
    #using Frames from RGBD camera (D435) (Uncomment the following line)
    ret,depth_img,color_img = camera.get_frame()

    #using Frames from any other camera (Uncomment the following lines)
    #ret, depth_img, color_img = camera.get_frame()
    # ret,frame = cap.read()

    #For handlandmark detection only (uncomment the following lines)
    hand_img = handDetect.find_hand(color_img)
    cv2.imshow("hand detected image", hand_img)

    #pose_img = poseDetect.find_pose(color_img)
    #cv2.imshow("pose detected image", pose_img)

    # holistic_img = holisticDetect.find_holistic(frame)

    
    #For holistic detection only (uncomment the following lines)
    # holistic_img = holisticDetect.find_holistic(color_img)
    # pose,leftHand, rightHand, face = holisticDetect.find_location()
    # cv2.imshow("pose detected image", drawBoundingBox(holistic_img,leftHand))



    #print(rightHand[4][0])
    #h,w,c = holistic_img.shape
    #print(int(rightHand[4][0]*w),int(rightHand[12][1]*h))
    #print(int(rightHand[20][0]*w),int(rightHand[0][1]*h))
    #drawBoundingBox(holistic_img,rightHand)
    
    #cv2.imshow("color image", color_img)
    #cv2.imshow("depth image", depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

 