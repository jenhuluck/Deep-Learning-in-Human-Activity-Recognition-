# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:09:38 2020

@author: Jieyun Hu
"""

# This file is for PAMAP2 data processing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
activities = {0: 'None', # 0: not part of the experiment. For instance the sensors are installed on the user or the user is performing activities unrelated to the experimental protocol, such as debriefing
              1: 'experiment', # 1: experiment, no freeze (can be any of stand, walk, turn)
              2: 'freeze' }

def read_files():
    
    files = os.listdir("./dataset")
    col_names = ["timestamp", "ankle_acc1", "ankle_acc2", "ankle_acc3", "upper_leg_acc1", "upper_leg_acc2","upper_leg_acc3", "Trunk_acc1", "Trunk_acc2", "Trunk_acc3", "annotation"]
    dataCollection = pd.DataFrame()
    for i, file in enumerate(files):
        print(file," is reading...")
        relative_file = "./dataset/"+file
        procData = pd.read_table(relative_file, header=None, sep='\s+')
        procData.columns = col_names
        procData['file_index'] = i # put the file index at the end of the row
        dataCollection = dataCollection.append(procData, ignore_index=True)       
        #break; # for testing short version, need to delete later       
    dataCollection.reset_index(drop=True, inplace=True)
    #print(dataCollection.shape)
    return dataCollection

def dataCleaning(dataCollection):
    dataCollection = dataCollection.drop(['timestamp'],axis = 1)  # removal of orientation columns as they are not needed
    dataCollection = dataCollection.drop(dataCollection[dataCollection.annotation == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
    dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
    print(dataCollection.isna().sum().sum())#count all NaN 
    print(dataCollection.shape)
    #dataCollection = dataCollection.dropna()
    #dataCollection = dataCollection.interpolate() 
    #removal of any remaining NaN value cells by constructing new data points in known set of data points
    #for i in range(0,4):
    #    dataCollection["heartrate"].iloc[i]=100 # only 4 cells are Nan value, change them manually
    print("data cleaned!")
    return dataCollection

def reset_label(dataCollection): 
    # Convert original labels {1, 2} to new labels. 
    mapping = {2:0,1:1} # old activity Id to new activity Id 
    for i in [2]:
        dataCollection.loc[dataCollection.annotation == i, 'annotation'] = mapping[i]

    return dataCollection

def segment(data, window_size): # data is numpy array
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][-2] == data[end][-2] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the same file
            X.append(data[start:(end+1),0:-2])
            y.append(data[start][-2]) 
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][-2] != data[start+1][-2]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}

def downsize(data):# data is numpy array
    downsample_size = 2
    data = data[::downsample_size,:]
    return data
'''
def scale_data(data):# data is numpy array
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
'''        
def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.')  

if __name__ == "__main__":
    file_name = 'Dap.h5'
    window_size = 25
    data = read_files()
    data = dataCleaning(data)
    data = reset_label(data) 
    #print(set(data.annotation))
    numpy_data = data.to_numpy()
    numpy_data = downsize(numpy_data) # downsize to 20%
    segment_data = segment(numpy_data, window_size) 
    save_data(segment_data, file_name)
    

    
    
