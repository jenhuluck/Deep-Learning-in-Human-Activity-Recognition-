# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:57:10 2020

@author: Jieyun Hu
"""

#split and images into training and testing set
#save into h5 format for modeling

import os
import h5py
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import natsort

#prepare training and testing dataset from images
#No time series
#reset label 8 to label 0
#Original labels are Null=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7, Subway=8
#New labels: Subway=0, Still=1, Walking=2, Run=3, Bike=4, Car=5, Bus=6, Train=7 
def prepare_dataset():
    paths = ["./picture1","./picture2","./picture3"]    
    X = []
    y = []
    for path in paths:
        dir = os.listdir(path)
        label_path = path + "/labels.txt"
        #print(label_path)
        np_labels = np.loadtxt(label_path)
        np_labels = np_labels.astype(np.int64)
        i = 0
        print(dir)
        for file in natsort.natsorted(dir): # order the files in ascending order 
            if file.endswith("jpg"): 
                print(file)
                file_path = path + "/" +file
                img = load_img(file_path,color_mode='grayscale')
                img = img_to_array(img).astype('float32')/255
                img = img.reshape(img.shape[0],img.shape[1],1)
                l = np_labels[i]
                if l != 0: # 0 is Null label, do not need for dataset
                    X.append(img)
                    if l == 8:                    
                        y.append(0) # reset label 8 to 0 
                    else:
                        y.append(l)
                i += 1
        
    print(set(y))
    X = np.asarray(X)
    y = np.asarray(y,dtype=int)
    print(X.shape)
    print(y.shape)
    return {'inputs' : X, 'labels': y}
    
def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.') 

if __name__ == "__main__":
    file_name = "video.h5"   
    data = prepare_dataset()
    save_data(data,file_name)
            
    