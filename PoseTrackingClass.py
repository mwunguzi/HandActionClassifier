# source for more details: https://google.github.io/mediapipe/solutions/pose.html
"""
//////////////////////////////////////////////////////////////////
||| Date: 03/09/2023                                          ///
||| Author: Herve Mwunguzi                                   ///
||| Purpose: Creating a module for mediapipe pose tracking  ///
|||///////////////////////////////////////////////////////////

"""
import cv2
import mediapipe as mp  

class PoseDetector():

    def __init__(self, static_img_mode = False, modelComplexity = 1, smoothLandmarks = True, enableSegmentation = False, 
                smoothSegmentation = True, minDetectConf = 0.5, minTrackingConf = 0.5):

        self.static_img_mode = static_img_mode # set to true for images and False for video stream input
        self.modelComplexity = modelComplexity # 0 or 1, defualt is 1
        self.smoothLandmarks = smoothLandmarks # set to true to reduce jitter
        self.enableSegmentation = enableSegmentation # set to true to generate segmantation mask
        self.smoothSegmentation = smoothSegmentation # set to true to reduce jitter
        self.minDetectConf = minDetectConf
        self.minTrackingConf = minTrackingConf

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose( self.static_img_mode, self.modelComplexity, self.smoothLandmarks, self.enableSegmentation, 
                                        self.smoothSegmentation, self.minDetectConf, self.minTrackingConf)
        self.mp_drawings = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_pose(self, img, draw = True):

        RGB_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting from BGR to RGB
        self.results = self.pose.process(RGB_img)

        # a "multi_hand_landmarks" field that contains the hand landmarks on each detected hand.
        if self.results.pose_landmarks: 
            
            if draw:
                self.mp_drawings.draw_landmarks(img,self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
                
        return img  


    def find_location(self, img, handNum=0, draw = True):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNum] # Obtaining all the lanmark from the preferred hand

            for id, landMark in enumerate(myHand.landmark):
                img_h, img_w, img_channel = img.shape
                channel_x, channel_y = int(landMark.x * img_w), int(landMark.y * img_h) # converting landmark to position coordinate
                lmList.append([id, channel_x, channel_y]) # append the pixal coordinate to the list
                cv2.circle(img, (channel_x, channel_y), 15, (255, 0, 255), cv2.FILLED)
        
        return lmList
        