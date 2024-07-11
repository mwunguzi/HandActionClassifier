# source for more details: https://google.github.io/mediapipe/solutions/pose.html
"""
//////////////////////////////////////////////////////////////////
||| Date: 03/09/2023                                          ///
||| Author: Herve Mwunguzi                                   ///
||| Purpose: Creating a module for mediapipe holistic tracking  ///
|||///////////////////////////////////////////////////////////

"""
import cv2
import mediapipe as mp
import numpy as np
  
class HolisticDetector():

    def __init__(self, static_img_mode = False, modelComplexity = 1, smoothLandmarks = True, enableSegmentation = False, 
                smoothSegmentation = True, refineFaceLandmarks = False, minDetectConf = 0.5, minTrackingConf = 0.5):

        self.static_img_mode = static_img_mode # set to true for images and False for video stream input
        self.modelComplexity = modelComplexity # 0 or 1, default is 1
        self.smoothLandmarks = smoothLandmarks # set to true to reduce jitter
        self.enableSegmentation = enableSegmentation # set to true to generate segmantation mask
        self.smoothSegmentation = smoothSegmentation # set to true to reduce jitter
        self.refineFaceLandmarks = refineFaceLandmarks 
        self.minDetectConf = minDetectConf
        self.minTrackingConf = minTrackingConf

        self.mp_holistic = mp.solutions.holistic 
        self.holistic = self.mp_holistic.Holistic( self.static_img_mode, self.modelComplexity, self.smoothLandmarks, self.enableSegmentation, 
                                        self.smoothSegmentation, self.refineFaceLandmarks, self.minDetectConf, self.minTrackingConf)
        self.mp_drawings = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_holistic(self, img, draw = True):

        RGB_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting from BGR to RGB
        self.results = self.holistic.process(RGB_img)

        # a "multi_hand_landmarks" field that contains the hand landmarks on each detected hand.
        if self.results.pose_landmarks: 
            
            if draw:
                #drawing the pose landmarks to the image
                self.mp_drawings.draw_landmarks(img,self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
                #drawing the left hand landmarks to the image
                self.mp_drawings.draw_landmarks(img,self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                                                                    self.mp_drawing_styles.get_default_hand_connections_style())
                #drawing the right hand landmarks to the image
                self.mp_drawings.draw_landmarks(img,self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                                                                    self.mp_drawing_styles.get_default_hand_connections_style())
                #drawing the face landmarks to the image
                self.mp_drawings.draw_landmarks(img,self.results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.mp_drawing_styles.get_default_face_mesh_tesselation_style(), 
                                                                                                    self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
        return img  


    def find_location(self, draw = True):

        pose = []
        leftHand = []
        rightHand = []
        face = []

        if self.results.pose_landmarks:

            myPose = self.results.pose_landmarks # Obtaining all the lanmark from the preferred hand

            for landMark in myPose.landmark:

                #print(myPose.landmark)
    
                pose.append(np.array([landMark.x, landMark.y, landMark.z, landMark.visibility])) # append the  to the list

        else: 
            pose.append(np.zeros((33,4))) # if no land mark detected append zeros to the array
                
        

        if self.results.left_hand_landmarks:

            myLeftHand = self.results.left_hand_landmarks # Obtaining all the lanmark from the preferred hand

            for landMark in myLeftHand.landmark:
                
                leftHand.append(np.array([landMark.x,landMark.y, landMark.z])) # append the  to the list

        else:
            leftHand.append(np.zeros((21,3))) # if no land mark detected append zeros to the array

        
        if self.results.right_hand_landmarks:

            myRightHand = self.results.right_hand_landmarks # Obtaining all the lanmark from the preferred hand

            for landMark in myRightHand.landmark:
                
                rightHand.append(np.array([landMark.x,landMark.y, landMark.z])) # append the  to the list

        else:
            rightHand.append(np.zeros((21,3))) # if no land mark detected append zeros to the array

        if self.results.face_landmarks:

            myface = self.results.face_landmarks # Obtaining all the lanmark from the preferred hand

            for landMark in myface.landmark:
                
                face.append(np.array([landMark.x,landMark.y, landMark.z])) # append the  to the list

        else:
            face.append(np.zeros((468,3))) # if no land mark detected append zeros to the array
        
        return pose,leftHand,rightHand,face