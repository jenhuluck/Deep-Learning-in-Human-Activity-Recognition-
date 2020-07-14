# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:41:56 2020

@author: Jieyun Hu
"""

'''
Apply different deep learning models on PAMAP2 dataset.
ANN,CNN and RNN were applied.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn import metrics
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, SimpleRNN, GRU, LSTM, GlobalMaxPooling1D,GlobalMaxPooling2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


class models():
    def __init__(self, path):
        self.path = path
       
    
    def read_h5(self):
        f = h5py.File(path, 'r')
        X = f.get('inputs')
        y = f.get('labels') 
       
        self.X = np.array(X)
        self.y = np.array(y)
        print(self.X[0][0])
        self.data_scale()
        print(self.X[0][0])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4, random_state = 11)
        
        print("X = ", self.X.shape)
        print("y =",self.y.shape)
              
        #print(self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape)
        #print(self.x_train[0].shape)
        #return X,y
    
    def data_scale(self):
        #Since sklearn scaler only allows scaling in 2d dim, so here first convert 3D to 2D. After scaling, convert 2D to 3D
        dim_0 = self.X.shape[0]
        dim_1 = self.X.shape[1]
        temp = self.X.reshape(dim_0,-1)
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaler.fit(temp)
        temp = scaler.transform(temp)
        self.X = temp.reshape(dim_0,dim_1,-1)
              
        
    def cnn_model(self):
        #K = len(set(self.y_train))
        K = 1
        #print(K)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test,-1)
        #print(self.x_train, self.x_test)
        i = Input(shape=self.x_train[0].shape)
        #Tested with several hidden layers. But since the it has only 9 features, it is not necessary to use multiple hidden layers
        x = Conv2D(32, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(i)
        x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        #x = Dropout(0.2)(x)
        #x = Conv2D(64, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.4)(x)
        #x = Conv2D(128, (3,3), strides = 2, activation = 'relu',padding='same',kernel_regularizer=regularizers.l2(0.0005))(x)
        #x = BatchNormalization()(x)
        #x = MaxPooling2D((2,2))(x)
        #x = Dropout(0.2)(x)
        x = Flatten()(x)    
        x = Dropout(0.2)(x)
        x = Dense(128,activation = 'relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K, activation = 'softmax')(x)       
        self.model = Model(i,x)
        self.model.compile(optimizer = Adam(lr=0.0005),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

        #self.r = model.fit(X, y, validation_split = 0.4, epochs = 50, batch_size = 32 )
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 64 )
        print(self.model.summary())
        # It is better than using keras do the splitting!!
        return self.r
    
    def dnn_model(self):
        K = 1
        i = Input(shape=self.x_train[0].shape)
        x = Flatten()(i)
        x = Dense(32,activation = 'relu')(x)
        #x = Dense(128,activation = 'relu')(x)
        #x = Dropout(0.2)(x)
        #x = Dense(256,activation = 'relu')(x)
        #x = Dropout(0.2)(x)
        #x = Dense(128,activation = 'relu')(x)
        x = Dense(K,activation = 'softmax')(x)
        self.model = Model(i,x)      
        self.model.compile(optimizer = Adam(lr=0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
        
        '''
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=self.x_train[0].shape),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(K,activation = 'softmax')
        ])
        model.compile(optimizer = Adam(lr=0.0005),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
        '''
        self.r = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        print(self.model.summary())
        return self.r
    

    def rnn_model(self):
        K = 1
        i = Input(shape = self.x_train[0].shape)
        x = LSTM(64)(i)
        x = Dense(32,activation = 'relu')(x)
        x = Dense(K,activation = 'softmax')(x)
        model = Model(i,x)      
        model.compile(optimizer = Adam(lr=0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
        self.r = model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs = 50, batch_size = 32 )
        print(self.model.summary())
        #self.r = model.fit(X, y, validation_split = 0.2, epochs = 10, batch_size = 32 )
        return self.r
   
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
    def con_matrix(self):
        K = len(set(self.y_train))
        self.y_pred = self.model.predict(self.x_test).argmax(axis=1)
        cm = confusion_matrix(self.y_test,self.y_pred)
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
    path = "./Dap.h5"
    dap = models(path)
    print("read h5 file....")
    dap.read_h5()
    
    if model_name == "cnn":
        dap.cnn_model()
    elif model_name == "dnn":
        dap.dnn_model()
    elif model_name == "rnn":
        dap.rnn_model()
    dap.draw()
    
    
    
