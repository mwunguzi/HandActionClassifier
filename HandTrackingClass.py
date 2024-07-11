# source for more details: https://google.github.io/mediapipe/getting_started/python.html
"""
//////////////////////////////////////////////////////////////////
||| Date: 03/09/2023                                          ///
||| Author: Herve Mwunguzi                                   ///
||| Purpose: Creating a module for mediapipe hand tracking  ///
|||///////////////////////////////////////////////////////////

"""
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hand=mp_hands.Hands()

class HandDetector():  

    def __init__(self, static_img_mode = False, maxNumHands = 2, modelComplexity = 1, minDetectConf = 0.5, minTrackingConf = 0.5):

        self.static_img_mode = static_img_mode # set to true for video stream and False for images
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity # 0 or 1, defualt is 1
        self.minDetectConf = minDetectConf
        self.minTrackingConf = minTrackingConf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands( self.static_img_mode, self.maxNumHands, self.modelComplexity, self.minDetectConf, self.minTrackingConf)
        self.mp_drawings = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hand(self, img, draw = True):

        RGB_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting from BGR to RGB
        self.results = self.hands.process(RGB_img)

        # a "multi_hand_landmarks" field that contains the hand landmarks on each detected hand.
        if self.results.multi_hand_landmarks: 
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawings.draw_landmarks(img,hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                                                                                                    self.mp_drawing_styles.get_default_hand_connections_style())
                
        return img  


    def find_location(self,handNum=0, draw = True):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNum] # Obtaining all the lanmark from the preferred hand

            for id, landMark in enumerate(myHand.landmark):
                #img_h, img_w, img_channel = img.shape
                #channel_x, channel_y = int(landMark.x * img_w), int(landMark.y * img_h) # converting landmark to position coordinate
                lmList.append([id, landMark.x, landMark.y]) # append the pixal coordinate to the list
                #cv2.circle(img, (, channel_y), 15, (255, 0, 255), cv2.FILLED)
        
        return lmList
        