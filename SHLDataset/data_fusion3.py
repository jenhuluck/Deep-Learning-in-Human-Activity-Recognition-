# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:38:36 2020

@author: Jieyun Hu
"""

#using deep learning on data fusion of motion and video data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn import metrics
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, SimpleRNN, GRU, LSTM, GlobalMaxPooling1D,GlobalMaxPooling2D,MaxPooling2D,BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.vis_utils import plot_model

class models():
    #def __init__(self):
       
    
    def read_h5(self, path_array):
        split_array = []
        l = len(path_array)
        for i, path in enumerate(path_array):
            f = h5py.File(path, 'r')
            X = f.get('inputs')
            y = f.get('labels') 

            X = np.array(X)
            y = np.array(y)
            split_array.append(X) # add X to array for split
            if i == l - 1:
                split_array.append(y) # add y to the last
          
        self.split = train_test_split(*split_array,test_size=0.2, random_state = 1)     
        '''
        print(len(split))
        print(split[0].shape) # data1_train_x
        print(split[1].shape) # data1_test_x
        print(split[2].shape) # data2_train_x
        print(split[3].shape) # data2_test_x
        print(split[4].shape) # y_train
        print(split[5].shape) # y_test
        '''
        return self.split
    
    # K is the number of classes
    def create_motion_cnn(self, input_shape, K):
        i = Input(shape = input_shape)
        x = Conv2D(16, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(i)
        x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        
        #x = Conv2D(32, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        #x = Dropout(0.2)(x)
        #x = Conv2D(256, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        #x = Conv2D(128, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        x = Flatten()(x)    
        x = Dropout(0.2)(x)
        x = Dense(128,activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(K,activation = 'relu')(x)
        model = Model(i, x)
        return model
    
    def create_img_cnn(self, input_shape, K):
        i = Input(shape = input_shape)
        x = Conv2D(32, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(64, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(0.4)(x)
        
        x = Conv2D(128, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        x = Dropout(0.5)(x)
        #x = Conv2D(128, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        x = Flatten()(x)    
        #x = Dropout(0.2)(x)
        x = Dense(256,activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(K,activation = 'relu')(x)
        model = Model(i, x)
        return model
    # merge n cnn models  
    def merge_models(self,n):
        motion_input_shape = np.expand_dims(self.split[0], -1)[0].shape
        K = len(set(self.split[-2]))
        print(motion_input_shape)
        cnns = [] # save all cnn models
        for i in range(n-1):
            cnn_i = self.create_motion_cnn(motion_input_shape,K)
            cnns.append(cnn_i)
        img_input_shape = np.expand_dims(self.split[-4], -1)[0].shape # last data should be image data
        print(img_input_shape)
        img_cnn = self.create_img_cnn(img_input_shape, K)
        cnns.append(img_cnn)
        #cnn1 = self.create_cnn(input_shape, K)
        #cnn2 = self.create_cnn(input_shape, K)
        #combinedInput = concatenate([cnn1.output, cnn2.output])
        combinedInput = concatenate([c.output for c in cnns])
        x = Dense(K,activation='softmax')(combinedInput)
        self.mix_model = Model(inputs = [c.input for c in cnns], outputs = x)
        #model = Model(inputs = [cnn1.input, cnn2.input], outputs = x)
        self.mix_model.compile(optimizer = Adam(lr=0.0005),loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
        #self.r = self.mix_model.fit(x = [np.expand_dims(self.split[0],-1),self.split[]])
        self.r = self.mix_model.fit(x = [np.expand_dims(self.split[i],-1) for i in range(2*n) if i % 2 == 0],
                           y = self.split[-2], validation_data = ([np.expand_dims(self.split[i],-1) for i in range(2*n) if i % 2 != 0],self.split[-1]), 
                           epochs = 50, batch_size = 256 )
        print(self.mix_model.summary())
        return self.r
    
        #r = model.fit(x = [np.expand_dims(self.split[0],-1),np.expand_dims(self.split[2],-1)], y = self.split[4], validation_data = ([np.expand_dims(self.split[1],-1),np.expand_dims(self.split[3],-1)],self.split[5]), epochs = 50, batch_size = 32 )
        
   
    def draw(self):
        f1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(self.r.history['loss'], label = 'loss')
        plt.plot(self.r.history['val_loss'], label = 'val_loss')
        plt.legend()
        f1.show()
        
        f2 = plt.figure(2)
        plt.plot(self.r.history['acc'], label = 'accuracy')
        plt.plot(self.r.history['val_acc'], label = 'val_accuracy')
        plt.legend()
        f2.show()
        
    # summary, confusion matrix and heatmap
    def con_matrix(self,n):
        K = len(set(self.split[-2]))
        self.y_pred = self.mix_model.predict([np.expand_dims(self.split[i],-1) for i in range(2*n) if i % 2 != 0]).argmax(axis=1)
        cm = confusion_matrix(self.split[-1],self.y_pred)
        self.plot_confusion_matrix(cm,list(range(K)))
        

    def plot_confusion_matrix(self, cm, classes, normalize = False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")
        print(cm)
        f3 = plt.figure(3)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd' 
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('predicted label')
            f3.show()
                


if __name__ == "__main__":
    model_name = "cnn" # can be cnn/dnn/rnn
    paths = ["./bag.h5","./image_for_fusion.h5"] # a motion data fuses with video data
    #paths = ["./bag.h5", "./hand.h5", "./hip.h5","./torso.h5", "./image_for_fusion.h5"]
    mix = models()
    print("read h5 file....")
    data_array = mix.read_h5(paths)
    mix.merge_models(len(paths))
    mix.draw()
    mix.con_matrix(len(paths))

