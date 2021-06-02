import cv2
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dropout,Dense,BatchNormalization,MaxPool2D,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Flatten
def create_dataset():
    va_path = "valence and arousal"
    va_parts = os.listdir(va_path)
    va_parts.remove('dataset_va_4.json')

    # x = []
    dataset={'img_arrays':[],'valence' : [], 'arousal' : []}

    # breakpoint()
    for part in va_parts:
        print(part)
        all_videos = os.listdir(os.path.join(va_path,part))
        # breakpoint()
        if os.path.exists(os.path.join(va_path,part,'README.md')):
            all_videos.remove('README.md')
        for video in all_videos:
            all_frames = os.listdir(os.path.join(va_path,part,video))
            all_frames.remove(video+'.json')
            for frame in all_frames:
                f = open(os.path.join("valence and arousal",part,video,video+'.json'),'r')
                data_va = json.load(f) 
                f.close()
                # frames = list(data_va["frames"].keys())
                # breakpoint()                        #valence and arousal\01\001\00000.png
                gray_img = cv2.imread(os.path.join(va_path,part,video,frame),cv2.IMREAD_GRAYSCALE)
                gray_img_resized = cv2.resize(gray_img,(64,64))
                dataset['img_arrays'].append(gray_img_resized.tolist())
                dataset['valence'].append(data_va['frames'][frame.split('.')[0]]['valence'])
                dataset['arousal'].append(data_va['frames'][frame.split('.')[0]]['arousal'])
    breakpoint()
    f = open("valence and arousal/dataset_va_all.json",'w')
    json.dump(dataset,f)

def load_dataset():
    f = open("valence and arousal/dataset_va_all.json",'r')
    dataset = json.load(f)
    x = dataset['img_arrays']
    y = [np.asarray([float(i),float(j)]) for i,j in zip(dataset['valence'],dataset['arousal'])]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True,random_state=1)
    x_train =np.asarray(x_train).reshape(-1,64,64,1)/255
    x_test = np.asarray(x_test).reshape(-1,64,64,1)/255
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    breakpoint()
    return x_train, x_test, y_train, y_test
load_dataset()

def neural_net():

    x_train, x_test, y_train, y_test = load_dataset()
    inp = Input(shape = (x_train[0].shape))
    conv1 = Conv2D(64,(4,4),2,activation='relu')(inp)
    batchnorm1 = BatchNormalization()(conv1)
    max_pool1 = MaxPool2D((2,2),padding='same')(batchnorm1)
    # conv2 = Conv2D(64,(4,4),2,activation='relu')(max_pool1)
    # batchnorm2 = BatchNormalization()(conv2)
    # # max_pool2 = MaxPool2D((2,2),padding='same')(batchnorm2)
    # conv3 = Conv2D(64,(4,4),2,activation='relu')(batchnorm2)
    # batchnorm3 = BatchNormalization()(conv3)
    # max_pool2 = MaxPool2D((2,2),padding='same')(batchnorm3)
    drop1= Dropout(0.3)(max_pool1)
    flat = Flatten()(drop1)
    dense = Dense(32,activation='relu')(flat)
    val = Dense(1,activation='relu',name='Valence')(dense)
    arousal = Dense(1,activation='relu',name='Arousal')(dense)
    model = Model(inputs=inp,outputs=[val,arousal])

    def scheduler(epoch,lr):
        if epoch%50==0:
            return lr*0.1
        else:
            return lr
    lr = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)
    adam = Adam(learning_rate=0.001)
    chkpoint = tensorflow.keras.callbacks.ModelCheckpoint("va model/weights2.h5",monitor='loss',save_best_only=True,mode='min')
    if os.path.exists("va model/weights2.h5"):
        model.load_weights("va model/weights2.h5")
        print("weights loaded")
    if not os.path.exists('va model/va_model2.json'):
        json_model = model.to_json()
        f = open('va model/va_model2.json','w')
        f.write(json_model)
        print("model_saved")
    model.compile(loss={'Valence':'mae','Arousal':'mae'}, metrics=['accuracy'],optimizer=adam)
    model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test),callbacks=[chkpoint,lr])
# neural_net()
# create_dataset()


