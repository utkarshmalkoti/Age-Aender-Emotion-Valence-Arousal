import json
import pickle
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential,save_model,model_from_json
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Input,Dropout,MaxPooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from facenet_pytorch import MTCNN
import mtcnn
import math

def mtcnn_detect_face(img,detector):
    faces = detector.detect_faces(img)
    return faces

def load_dataset_age_gen():
    with open("resized_dataset_age_gen.pickle",'rb') as f:
        img = pickle.load(f)
    x = np.array(img['pixel_array'])  #/255
    y = np.array([[int(i),int(j)] for i,j in zip(img['gens'],img['ages'])])
    x_train_ag,x_test_ag,y_train_ag,y_test_ag = train_test_split(x,y,test_size = 0.2,random_state=1)
    return x_train_ag,x_test_ag,y_train_ag,y_test_ag

def load_dataset_emotion():
    train_data = {}
    test_data = {}
    with open("facial expression affectnet\dataset_facial_exp_gray_5k_train.pickle",'rb') as f:
        train_data = pickle.load(f)
    with open("facial expression affectnet\dataset_facial_exp_gray_5k_test.pickle",'rb') as f:
        test_data = pickle.load(f)
    x_train = np.asarray(train_data[1])
    y_train = to_categorical(np.asarray(train_data[0]))
    x_test = np.asarray(test_data[1])
    y_test = to_categorical(np.asarray(test_data[0]))
    return x_train,y_train,x_test,y_test

def neural_network():
    # AGE GENDER MODEL LOAD
    json_file = open('models/age_gender_model/model.json','r')
    json_model = json_file.read()
    json_file.close()
    model_ag = model_from_json(json_model)
    model_ag.load_weights('models/age_gender_model/age_gender_weights.h5')
    
    # EMOTION MODEL LOAD
    json_file_emo = open('models/emotions/kaggle_model_3_2conv.json','r')
    json_model_emo = json_file_emo.read()
    json_file_emo.close()
    model_emotion = model_from_json(json_model_emo)
    model_emotion.load_weights('models/emotions/kaggle_emotion_weights_3_2conv.h5')

    # VALENCE AND AROUSAL MODEL LOAD
    json_file_va = open('models/va model/va_model2.json','r')
    json_model_va = json_file_va.read()
    json_file_va.close()
    model_va = model_from_json(json_model_va)
    model_va.load_weights('models/va model/weights2.h5')

    #COMPILE ALL MODELS
    adam = Adam(learning_rate=1e-3)
    model_va.compile(loss=["categorical_crossentropy"],metrics=['accuracy'],optimizer=adam)
    model_emotion.compile(loss={'Valence':'mae','Arousal':'mae'}, metrics=['accuracy'],optimizer=adam)
    model_ag.compile(loss={'gen':"binary_crossentropy",'age':"mae"}, metrics=['accuracy'],optimizer = adam)

    return [model_ag,model_va,model_emotion]


lables = {"Anger":0,"Disgust":1,"Fear":2,"Happy":3,"Neutral":4,"Sad":5,"Surprise":6} 
face_metric = {"FaceAnalyzed":False, 
                "FacialExpression":{
                    "DominantBasicEmotion":"",
                    "BasicEmotions":{
                        "Angry":0,
                        "Disgust":0,
                        "Fear":0,
                        "Happy":0,
                        "Neutral":0,
                        "Sad":0,
                        "Surprise":0
                    },
                    "Valence":0,
                    "Arousal":0
                },
                "Characteristics":{
                    "Gender": "",
                    "Age": 0
                },
                "BoundingBox":{
                    "Left":0,
                    "Top":0,
                    "Width":0,
                    "Height":0
                },
                "NumberOfFaces":0
                }
img_add=str(input("Enter the address: "))
img= cv2.imread(img_add)
detector = mtcnn.MTCNN()
while(img_add!='-1'):
    # faces_imgs = detect_face(img_add)
    faces = mtcnn_detect_face(img,detector)
    # for i in range(0,faces_imgs['No_of_faces']):
    for i in range(0,len(faces)):
        img = cv2.imread(img_add) 
        box = faces[i]['box']
        face = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0),thickness=2)
        face_metric['FaceAnalyzed']=True
        # gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #PREPARE INPUT
        gray_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face,(64,64))
        final_input = np.asarray(resized_face).reshape(-1,64,64,1)/255
        model_ag,model_va,model_emotion = neural_network()
        
        #Age_gender_output
        out_ag = model_ag.predict(final_input)
        if out_ag[0]>0.5:
            face_metric['Characteristics']['Gender']='Female'
        else:
            face_metric['Characteristics']['Gender']='Male'
        face_metric['Characteristics']['Age']=out_ag[1][0][0]
        
        #Emotion_output
        out_emo = model_emotion.predict(final_input)
        keys = [i for i in lables.keys()]
        values = [i for i in lables.values()]
        face_metric['FacialExpression']['DominantBasicEmotion'] = keys[values.index(np.argmax(out_emo))]
        face_metric['FacialExpression']['BasicEmotions']['Angry']=out_emo[0][0]
        # face_metric['FacialExpression']['BasicEmotions']['Contempt']=out_emo[0][1]
        face_metric['FacialExpression']['BasicEmotions']['Disgust']=out_emo[0][1]
        face_metric['FacialExpression']['BasicEmotions']['Fear']=out_emo[0][2]
        face_metric['FacialExpression']['BasicEmotions']['Happy']=out_emo[0][3]
        face_metric['FacialExpression']['BasicEmotions']['Neutral']=out_emo[0][4]
        face_metric['FacialExpression']['BasicEmotions']['Sad']=out_emo[0][5]
        face_metric['FacialExpression']['BasicEmotions']['Surprise']=out_emo[0][6]
        #Valence_arousal_output
        out_va = model_va.predict(final_input)
        face_metric['FacialExpression']['Valence']=out_va[0][0][0]
        face_metric['FacialExpression']['Arousal']=out_va[1][0][0]
        #BoundingBox
        face_metric['BoundingBox']['Left'] = box[0]
        face_metric['BoundingBox']['Top'] = box[1]
        face_metric['BoundingBox']['Width'] = box[2]
        face_metric['BoundingBox']['Height'] = box[3]
        
        #No_of_faces
        face_metric['NumberOfFaces'] = len(faces)
        print(face_metric,"\n")
    img_add=str(input("Enter the address: "))
