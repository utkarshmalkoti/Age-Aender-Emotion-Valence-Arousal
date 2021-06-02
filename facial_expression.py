import os
import cv2
import tensorflow as tf
import json
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,BatchNormalization,Dropout,Flatten,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.python.keras.callbacks import EarlyStopping




lables = {"angry":0,"disgust":1,"fear":2,"happy":3,"neutral":4,"sad":5,"surprise":6} #"contempt":1,
dataset_type = ["test","train"]
all_file_path = "Datasets\kaggle_emo\images"


def create_dataset():
    for dataset in dataset_type:
        img_lables=[]
        img_arrays=[]
        print(dataset)
        dataset_path = os.path.join(all_file_path,dataset)  #facial expression affnet/train
        print(dataset_path)
        for exp in lables.keys():
            print(exp,os.path.join(dataset_path,exp))
            for i,img in enumerate(os.listdir(os.path.join(dataset_path,exp))):    #facial expression affnet/train/Anger
                if dataset == 'train' and i<5000:
                    img_array = cv2.imread(filename=os.path.join(dataset_path,exp,img))  #facial expression affectnet\test\Anger16\18.jpg
                    gray_img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
                    img_array_resized = cv2.resize(gray_img_array,(64,64))
                    img_lables.append(lables[exp])
                    img_arrays.append(img_array_resized)
                elif dataset=='test' and i<1500:
                    img_array = cv2.imread(filename=os.path.join(dataset_path,exp,img))  #facial expression affectnet\test\Anger16\18.jpg
                    gray_img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
                    img_array_resized = cv2.resize(gray_img_array,(64,64))
                    img_lables.append(lables[exp])
                    img_arrays.append(img_array_resized)
                else:
                    break
        f= open(f"Datasets\kaggle_emo\images\kaggle_{dataset}.pickle",'wb') 
        pickle.dump([img_lables,img_arrays],f)

def data_augmentation():
    datagen = ImageDataGenerator(rotation_range=40,zoom_range=0.2,horizontal_flip=True,shear_range=0.2,fill_mode='nearest')
    data_mul={"Anger":0,"Contempt":20,"Disgust":10,"Fear":6,"Happy":0,"Neutral":0,"Sad":2,"Surprise":3}
    for dataset in dataset_type:
        dataset_path = os.path.join(all_file_path,dataset) 
        for exp in lables:
            if(data_mul[exp]!=0):
                for no,img in enumerate(os.listdir(os.path.join(dataset_path,exp))): 
                    img_array = cv2.imread(filename=os.path.join(dataset_path,exp,img))
                    rgb_img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                    rgb_img_array = rgb_img_array.reshape(1,224,224,3)
                    if(no%1000==0):
                        print(img,"   ",no)
                    i=0
                    for batch in datagen.flow(rgb_img_array,save_to_dir=os.path.join(all_file_path,'aug'),save_prefix='aug_again'+exp,save_format="jpg",batch_size=1):
                        i+=1
                        if(i==data_mul[exp]):
                            break

def load_dataset():
    train_data = {}
    test_data = {}
    with open("Datasets/kaggle_emo/images/kaggle_train.pickle",'rb') as f:
        train_data = pickle.load(f)
    with open("Datasets/kaggle_emo/images/kaggle_test.pickle",'rb') as f:
        test_data = pickle.load(f)
    x_train = np.asarray(train_data[1])
    y_train = to_categorical(np.asarray(train_data[0]))
    x_test = np.asarray(test_data[1])
    y_test = to_categorical(np.asarray(test_data[0]))
    return x_train,y_train,x_test,y_test

def neural_network():
    x_train,y_train,x_test,y_test = load_dataset()
    x_train = np.reshape(x_train,(-1,64,64,1))/255
    x_test = np.reshape(x_test,(-1,64,64,1))/255
    if(not os.path.exists("Datasets\kaggle_emo\kaggle_model_drive_3_2conv.json")):
        with open("Datasets\kaggle_emo\kaggle_model_drive_3_2conv.json",'r') as f:
            json_model = f.read()
        model = model_from_json(json_model)
        print("model_Loaded")
        model.compile(loss=["categorical_crossentropy"],metrics=['accuracy'],optimizer='adam')
        model.load_weights("Datasets/kaggle_emo/kaggle_emotion_weights_drive_3_2conv.h5")    
        print("weight loaded")
        model.evaluate(x_test,y_test)
        model.fit(x_train,y_train,batch_size=50,epochs=10,validation_data=(x_test,y_test))
    else:
        model = Sequential()
        #1
        model.add(Conv2D(64,4,(2,2),input_shape=(64,64,1),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #2
        model.add(Conv2D(128,4,(2,2),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2),padding='same'))
        #3
        model.add(Conv2D(128,4,(2,2),padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2),padding='same'))

        model.add(Flatten())
    
        model.add(Dense(64))
        model.add(Activation('relu')) 

        model.add(Dropout(0.3))

        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dense(7))
        model.add(Activation('softmax'))
        
        file = "3*2conv"

        def scheduler(epoch,learning_rate):
            if epoch%20==0:
                return learning_rate*0.75
            else:
                return learning_rate
        lr = tensorflow.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
        adam = Adam(learning_rate=1e-2)
        earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=3,mode='min',restore_best_weights=True,verbose=1)
        checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(f"Datasets/kaggle_emo/images/kaggle_emotion_weights_{file}.h5",monitor="val_loss",save_best_only=True,mode='min')        
    
        if os.path.exists(f"Datasets\kaggle_emo\kaggle_emotion_weights_drive_{file}.h5"):
            model.load_weights("Datasets\kaggle_emo\kaggle_emotion_weights_drive_{file}}.h5")    
            print("weight loaded")
            # model.load_weights(f"Datasets\kaggle_emo\kaggle_emotion_weights_drive_3_2conv.h5")
        model.compile(loss=["categorical_crossentropy"],metrics=['accuracy'],optimizer=adam)
        model.fit(x_train,y_train,batch_size=50,epochs=50,validation_data=(x_test,y_test),callbacks=[lr,checkpoint,earlystop])

        json_model = model.to_json()
        with open(f"Datasets/kaggle_emo/images/kaggle_model{file}.json",'w') as f:
            f.write(json_model)
    return model
neural_network()
