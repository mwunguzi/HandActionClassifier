from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from holisticTrackingClass import *
import numpy as np
import cv2
import csv
import os   
import sys

plt.style.use("ggplot")
plt.figure(figsize=(10,10))
maxInt = sys.maxsize

def show_shapes(): # can make yours to take inputs; this'll use local variable values
    print("Expected: (num_samples, timesteps, channels)")
    print("Sequences: {}".format(Sequences.shape))
    print("Targets:   {}".format(Targets.shape))  

def vidLength(file_path):

    data = cv2.VideoCapture(file_path)
  
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    
    # calculate duration of the video
    seconds = frames // fps
    return seconds

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

holisticDetect =HolisticDetector(False,1,True,False,True,False,0.5,0.5)

filePath = os.path.join('gesturesDataset')
#controlActions = np.array(["Takeover"])
controlActions = np.array(["Reverse","Forward","TurnLeft", "TurnRight","Stop","Takeover","Release"])
numVideos = 30 #number of videos for each action
numFrames = 30 #number of frames in one video 

#Reverse:0, Forward:1, TurnLeft:2, TurnRight:3, Stop:4, Takeover:5, Release:6
label_map = {label:num for num, label in enumerate(controlActions)} # creating label map

#loading the keypoint frorm the training dataset
data,labels = [],[]

#for training with videos uncomment the following lines
'''
for action in controlActions:
     
     directory = os.path.join('TrainingVideosSet3',action) #creating path to the training videos

     for filename in os.listdir(directory): # iterating through the files
          
          f = os.path.join(directory,filename) #getting file 

          if os.path.isfile(f):

            cap = cv2.VideoCapture(f)
            print(f)

            duration=vidLength(f)
            print("duration: {}".format(duration))
            

            window = []
            
            frame_count = 0
            fps_threshold = 28
            duration_threshold = 1

            counts = duration // duration_threshold
            print("count: {}".format(counts))

            while True:
                #ret, depth_img, color_img = camera.get_frame()
                ret,frame = cap.read()
                #hand_img = handDetect.find_hand(color_img)
                #cv2.imshow("hand detected image", hand_img)

                #pose_img = poseDetect.find_pose(color_img)
                #cv2.imshow("pose detected image", pose_img)
                if ret == True:
                    frame_count +=1
                    #print(frame_count)
                    holistic_img = holisticDetect.find_holistic(frame)
                    pose,leftHand, rightHand, face = holisticDetect.find_location()
                    cv2.imshow("pose detected image", holistic_img)
                    
                    if frame_count <= fps_threshold: 
                        #converting the list to numpy array and flattened for one dimension
                        pose = np.array(pose).flatten()
                        leftHand = np.array(leftHand).flatten()
                        rightHand = np.array(rightHand).flatten()
                        face = np.array(face).flatten()
                        #allKeypoints = np.concatenate([pose,leftHand,rightHand,face])
                        # allKeypoints = np.concatenate([pose,leftHand,rightHand])
                        allKeypoints = np.concatenate([leftHand,rightHand])
                        window.append(allKeypoints)

                    elif frame_count > fps_threshold:
                        
                        if counts == 1:
                            data.append(window)
                            labels.append(label_map[action])
                            #print(counts)
                            break
                        elif counts > 1:
                            data.append(window)
                            labels.append(label_map[action])
                            frame_count = 0
                            window.clear()
                            counts -=1
                            #print(counts)


                #cv2.imshow("color image", color_img)
                #cv2.imshow("depth image", depth_img)
                elif ret == False :
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            x = np.array(data)
            print(x.shape)
'''                      

print("Finished Loading data")

'''
with open("data2.csv", "w", newline="") as f:
   writer = csv.writer(f)
   writer.writerows(data)

with open("label2.csv", "w", newline="") as f:
   writer = csv.writer(f)
   writer.writerow(labels)





with open("data2.csv", "r") as file2:
    rawdata2 = file2.readlines()

    for row in rawdata2:
        data2=row.split(",")

file2 = open("label.csv", "r")
labels = np.loadtxt(csv.reader(file2))
file2.close()


file = open("data2.csv", "r")
data2 = list(csv.reader(file))
file.close()


with open("label2.csv", "r") as file:
    rawdata = file.readlines()

    for c in rawdata:
        labels2=c.split(",")

'''

#x2 = np.array(data2)

#for testing uncomment these lines
x = np.load("data.npy")
labels = np.load("labels.npy")
print(x.shape)

# uncomment the following line when trainning 

# x = np.array(data)
# np.save("data.npy",x)
# np.save("labels.npy",labels)
# print(x.shape)


# print("Finished converting frames into npy array")

# xData = np.load("data.npy")
# print(xData.shape)
# print(x2.shape)
# print(xData.shape)
# print(len(x))
# print(len(labels2))




y = to_categorical(labels).astype(int) #converting to onehot encoded
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) #splittingdata for train and testing

print("checking x-trian and y-train")
print(type(x_train))
print(type(y_train))

Sequences = np.asarray(x_train)
Targets   = np.asarray(y_train)
show_shapes()

#creating log files for visualization  from Tensor boards(dashboard)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

l2=regularizers.L2(1e-5)
l1l2=regularizers.L1L2(l1=1e-4, l2=1e-5)

# # model with 1 LSTM Layer
# model1 = Sequential()
# model1.add(LSTM(128,return_sequences=True,kernel_regularizer= l2, activation='elu', input_shape=(28,126)))
# #model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(40,1662)))
# #model.add(LSTM(64,return_sequences=True,kernel_regularizer= l2,  activation='elu'))
# # model1.add(LSTM(32,return_sequences=False,kernel_regularizer= l2, activation='elu'))
# #model.add(Dense(64,kernel_regularizer= l2, activation='elu'))
# model1.add(Dense(32,kernel_regularizer= l2, activation='elu'))
# model1.add(Dense(controlActions.shape[0], activation='softmax'))

# model1.summary()
# adam =Adam(learning_rate=0.00001)
# model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# print("Starting Training ...")
# history = model1.fit(x_train, y_train, epochs=40,  validation_data = (x_test,y_test) ,callbacks=[tb_callback])
# model1.save('GestureRecognition_1layers.h5')

# plt.plot(history.history["accuracy"], label="2Layers_Train_accuracy  ")
# plt.plot(history.history["val_accuracy"], label="2Layers_Test_accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.ylim([0.0, 1.2])
# plt.legend(loc="lower right")
# plt.savefig("train_val_acc_1layers_1Ex5.png")


# test_loss, test_acc = model1.evaluate(x_test, y_test, verbose=2)

# yhat = model1.predict(x_test)

# ytrue = np.argmax(y_test,axis=1).tolist()
# yhat = np.argmax(yhat,axis=1).tolist()

# print("Accuracy metrics for  1 layer LSTM")
# print(multilabel_confusion_matrix(ytrue,yhat))
# print(accuracy_score(ytrue,yhat))






# model with 2 LSTM Layer
model2 = Sequential()
model2.add(LSTM(128,return_sequences=True,kernel_regularizer= l2, activation='elu', input_shape=(28,126)))
#model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(40,1662)))
#model.add(LSTM(64,return_sequences=True,kernel_regularizer= l2,  activation='elu'))
model2.add(LSTM(32,return_sequences=False,kernel_regularizer= l2, activation='elu'))
#model.add(Dense(64,kernel_regularizer= l2, activation='elu'))
model2.add(Dense(32,kernel_regularizer= l2, activation='elu'))
model2.add(Dense(controlActions.shape[0], activation='softmax'))

model2.summary()
adam =Adam(learning_rate=0.00001)
model2.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting Training ...")
history = model2.fit(x_train, y_train, epochs=40,  validation_data = (x_test,y_test) ,callbacks=[tb_callback])
model2.save('GestureRecognition_2layers.h5')

fig, (axis1,axis2,axis3) = plt.subplots(nrows=3,ncols=1,sharex=True)
axis1.set_title("Accuracy for hand gesture detection using LSTM")
axis1.plot(history.history["categorical_accuracy"], label="2Layers_accuracy  ")
axis1.plot(history.history["val_categorical_accuracy"], label="2Layers_val_accuracy")
axis1.set_ylabel("Accuracy")
axis1.set_ylim([0.0, 1.2])
axis1.legend(loc="lower right")



test_loss, test_acc = model2.evaluate(x_test, y_test, verbose=2)

yhat = model2.predict(x_test)

ytrue = np.argmax(y_test,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()
print("Accuracy metrics for  2 layer LSTM")
print(multilabel_confusion_matrix(ytrue,yhat))
print(accuracy_score(ytrue,yhat))
print({name: accuracy_score(np.array(ytrue) == i, np.array(yhat) == i) for i, name in enumerate(controlActions)})
axis1.text(0,0.8,'Accuracy = %.2f' %accuracy_score(ytrue,yhat))


# model with 3 LSTM Layer
model3 = Sequential()
model3.add(LSTM(128,return_sequences=True,kernel_regularizer= l2, activation='elu', input_shape=(28,126)))
#model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(40,1662)))
model3.add(LSTM(64,return_sequences=True,kernel_regularizer= l2,  activation='elu'))
model3.add(LSTM(32,return_sequences=False,kernel_regularizer= l2, activation='elu'))
#model.add(Dense(64,kernel_regularizer= l2, activation='elu'))
model3.add(Dense(32,kernel_regularizer= l2, activation='elu'))
model3.add(Dense(controlActions.shape[0], activation='softmax'))

model3.summary()
adam =Adam(learning_rate=0.00001)
model3.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting Training ...")
history = model3.fit(x_train, y_train, epochs=40,  validation_data = (x_test,y_test) ,callbacks=[tb_callback])
model3.save('GestureRecognition_3layers.h5')

axis2.plot(history.history["categorical_accuracy"], label="3Layers_accuracy  ")
axis2.plot(history.history["val_categorical_accuracy"], label="3Layers_val_accuracy")
axis2.set_ylabel("Accuracy")
axis2.set_ylim([0.0, 1.2])
axis2.legend(loc="lower right")



test_loss, test_acc = model3.evaluate(x_test, y_test, verbose=2)

yhat = model3.predict(x_test)

ytrue = np.argmax(y_test,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()
print("Accuracy metrics for  3 layer LSTM")
print(multilabel_confusion_matrix(ytrue,yhat))
print(accuracy_score(ytrue,yhat))
print({name: accuracy_score(np.array(ytrue) == i, np.array(yhat) == i) for i, name in enumerate(controlActions)})
axis2.text(0,0.8,'Accuracy = %.2f' %accuracy_score(ytrue,yhat))



#model with 4 LSTM layer
model4 = Sequential()
model4.add(LSTM(128,return_sequences=True,kernel_regularizer= l2, activation='elu', input_shape=(28,126)))
#model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(40,1662)))
model4.add(LSTM(64,return_sequences=True,kernel_regularizer= l2,  activation='elu'))
model4.add(LSTM(32,return_sequences=True,kernel_regularizer= l2, activation='elu'))
model4.add(LSTM(16,return_sequences=False,kernel_regularizer= l2, activation='elu'))
#model.add(Dense(64,kernel_regularizer= l2, activation='elu'))
model4.add(Dense(16,kernel_regularizer= l2, activation='elu'))
model4.add(Dense(controlActions.shape[0], activation='softmax'))

model4.summary()
adam2 =Adam(learning_rate=0.00001)
model4.compile(optimizer=adam2, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting Training ...")
history = model4.fit(x_train, y_train, epochs=40,  validation_data = (x_test,y_test) ,callbacks=[tb_callback])
model4.save('GestureRecognition_4layers.h5')

axis3.plot(history.history["categorical_accuracy"], label="4Layers_accuracy  ")
axis3.plot(history.history["val_categorical_accuracy"], label="4Layers_val_accuracy")
axis3.set_xlabel("Epoch")
axis3.set_ylabel("Accuracy")
axis3.set_ylim([0.0, 1.2])
axis3.legend(loc="lower right")



test_loss, test_acc = model4.evaluate(x_test, y_test, verbose=2)

yhat = model2.predict(x_test)

ytrue = np.argmax(y_test,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()
print("Accuracy metrics for  4 layer LSTM")
print(multilabel_confusion_matrix(ytrue,yhat))
print(accuracy_score(ytrue,yhat))
print({name: accuracy_score(np.array(ytrue) == i, np.array(yhat) == i) for i, name in enumerate(controlActions)})
axis3.text(0,0.8,'Accuracy = %.2f' %accuracy_score(ytrue,yhat))

plt.savefig("HandLandmark_1Ex5.png",dpi=200)