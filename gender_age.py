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

def neural_network():
    if os.path.exists('models/ge_gender_model/model.json'):
        # AGE GENDER MODEL LOAD
        json_file = open('models/age_gender_model/model.json','r')
        json_model = json_file.read()
        json_file.close()
        model_ag = model_from_json(json_model)
        model_ag.load_weights('models/age_gender_model/age_gender_weights.h5')     
        model.compile(loss={'gen':"binary_crossentropy",'age':"mae"},optimizer = 'adam', metrics=['accuracy'])
    else:
        x_train,x_test,y_train,y_test = load_dataset_age_gen()
        x_train = np.asarray(x_train).reshape(-1,64,64,1)/255
        x_test = np.asarray(x_test).reshape(-1,64,64,1)/255
        y_train = np.asarray(y_train).reshape(-1,2)
        y_test = np.asarray(y_test).reshape(-1,2)
        input_layer = Input((x_train[0].shape))
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
        model = Model(inputs=input_layer, outputs =[gen_out,age_out])
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        def scheduler(epoch, lr):
            if epoch%15==0:
                return lr*0.1
            else:
                print(lr)
                return lr
        learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        checkpoint = tf.keras.callbacks.ModelCheckpoint("models/age_gender_model/age_gender_weights.h5", monitor= 'loss', save_best_only=True, mode='min')
        adam = Adam(learning_rate=1e-3)
        model_emotion.compile(loss=["categorical_crossentropy"],metrics=['accuracy'],optimizer=adam)
        model.compile(loss={'gen':"binary_crossentropy",'age':"mae"},optimizer = adam, metrics=['accuracy'])
        model.load_weights("age_gender_model/age_gender_weights.h5")
        model.fit(x_train_ag,[y_train_ag[:,0],y_train_ag[:,1]],batch_size=50,validation_data = (x_test_ag,[y_test_ag[:,0],y_test_ag[:,1]]),epochs=1, callbacks=[learning_rate,checkpoint])


