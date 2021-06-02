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

lables = {"Anger":0,"Contempt":1,"Disgust":2,"Fear":3,"Happy":4,"Neutral":5,"Sad":6,"Surprise":7} 


def detect_face(img_add):
    # img = cv2.imread(img_add)
    gray_img = cv2.imread(img_add,cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img,(512,512))
    profile_cascade = cv2.CascadeClassifier("Cascade/data/haarcascade_profileface.xml")
    frontal_cascade = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt2.xml')
    frontal_cascade_default = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_default.xml')
    frontal_cascade_alt = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt.xml')
    frontal_cascade_cat = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalcatface.xml')
    frontal_cascade_cat_ext = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalcatface_extended.xml')
    frontal_cascade_tree = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt_tree.xml')

    face = frontal_cascade.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = profile_cascade.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = frontal_cascade_default.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = frontal_cascade_alt.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = frontal_cascade_cat.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = frontal_cascade_tree.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)
    if face==():
        face = frontal_cascade_cat_ext.detectMultiScale(gray_img,scaleFactor=1.5,minNeighbors=3)

    face_img = []
    for (x,y,w,h) in face:
        cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),2)
        face_img.append(gray_img[x:x+w+22,y:y+h+22])
    
    return face_img




def build_dataset(): 
    age_gen_path_imgs = "age, gender/UTKFace"
    age_gen_imgs_names = os.listdir(age_gen_path_imgs)

    img = {'ages' : [],'gens' : [],'pixel_array' : []}

    for i in age_gen_imgs_names:
        img['ages'].append(int(i.split('_')[0]))
        img['gens'].append(int(i.split('_')[1]))
        gray_img = cv2.imread(age_gen_path_imgs+'/'+i,cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(gray_img,(64,64))
        img['pixel_array'].append(resized_img)
    

    with open("resized_dataset_age_gen.pickle",'wb') as f:
        pickle.dump(img,f)


def load_dataset_age_gen():
    if os.path.exists('resized_dataset_age_gen.pickle')==False:
        build_dataset()
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
    if os.path.exists('age_gender_model/model.json'):
        # AGE GENDER MODEL LOAD
        json_file = open('age_gender_model/model.json','r')
        json_model = json_file.read()
        json_file.close()
        model_ag = model_from_json(json_model)
        model_ag.load_weights('age_gender_model/age_gender_weights.h5')
        # EMOTION MODEL LOAD
        json_file_emo = open('emotion_model/model.json','r')
        json_model_emo = json_file_emo.read()
        json_file_emo.close()
        model_emotion = model_from_json(json_model_emo)
        model_emotion.load_weights('emotion_model/emotion_weights_5k.h5')
        
    else:
        x_train_ag,x_test_ag,y_train_ag,y_test_ag = load_dataset_age_gen()
        x_train_ag = np.asarray(x_train_ag).reshape(-1,64,64,1)/255
        x_test_ag = np.asarray(x_test_ag).reshape(-1,64,64,1)/255
        y_train_ag = np.asarray(y_train_ag).reshape(-1,2)
        y_test_ag = np.asarray(y_test_ag).reshape(-1,2)
        input_layer = Input((x_train_ag[0].shape))
        conv1 = Conv2D(64,(4,4),activation="relu",strides=(2,2))(input_layer)
        batchnorm1 = BatchNormalization()(conv1)
        maxpool1 = MaxPooling2D((2,2))(batchnorm1)
        drop1 = Dropout(0.3)(maxpool1)
        # conv2 = Conv2D(64,(2,2), activation='relu',strides=(1,1))(drop1)
        # dense = Dense(32,activation = 'relu')(conv2)
        # conv3 = Conv2D(64,(4,4),activation="relu",strides=(2,2))(drop1)
        # batchnorm3 = BatchNormalization()(conv3)
        # maxpool3 = MaxPooling2D((2,2))(batchnorm3)
        # drop3 = Dropout(0.3)(maxpool3)
        flat = Flatten()(drop1)
        dense1 = Dense(32,activation = 'relu')(flat)
        # dense2 = Dense(32,activation = 'relu')(flat)
        age_out = Dense(1,activation = 'relu',name = 'age')(dense1)#(dense1)#
        gen_out = Dense(1,activation= 'sigmoid', name = 'gen')(flat)#(dense2)#
        # merged = tf.reshape(tf.stack([gen_out, age_out], 2),(-1,))
        # tf.keras.layers.concatenate([gen_out,age_out], axis=-1)
        model_ag = Model(inputs=input_layer, outputs =[gen_out,age_out])
        model_json = model_ag.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
    def scheduler(epoch, lr):
        if epoch%15==0:
            return lr*0.1
        else:
            print(lr)
            return lr
    learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint = tf.keras.callbacks.ModelCheckpoint("age_gender_model/age_gender_weights.h5", monitor= 'loss', save_best_only=True, mode='min')
    adam = Adam(learning_rate=1e-3)
    model_emotion.compile(loss=["categorical_crossentropy"],metrics=['accuracy'],optimizer=adam)
    model_ag.compile(loss={'gen':"binary_crossentropy",'age':"mae"},optimizer = adam, metrics=['accuracy'])
    model_ag.load_weights("age_gender_model/age_gender_weights.h5")
    # model_ag.fit(x_train_ag,[y_train_ag[:,0],y_train_ag[:,1]],batch_size=50,validation_data = (x_test_ag,[y_test_ag[:,0],y_test_ag[:,1]]),epochs=1, callbacks=[learning_rate,checkpoint])
    
    # model_ag.evaluate()
    # model_emotion.evaluate()
    return model_ag,model_emotion


# img_add = "C:/Users/utkarsh malkoti/Desktop/Utkarsh/mypic.jpg"
img_add = "try.jpg"
faces = detect_face(img_add)
for face in faces:
    # gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    resized_face = cv2.resize(face,(64,64))
    final_input = np.asarray(resized_face).reshape(-1,64,64,1)/255
    model_ag,model_emotion = neural_network()
    out_ag = model_ag.predict(final_input)
    out_emo = model_emotion.predict(final_input)
    if out_ag[0]>0.5:
        print("gender = Female",out_ag[0])
    else:
        print("gender = Male",out_ag[0])
    print("age = ",out_ag[1])
    keys = list(lables.keys())
    values = list(lables.values())
    print("emotion = ", out_emo)#keys[values.index[out_emo.argmax()]])




