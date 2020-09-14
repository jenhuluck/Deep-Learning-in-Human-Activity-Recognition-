# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:49:36 2020

@author: Jieyun Hu
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
import os
import csv
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#data fusion of video data, acceleration data and labels.
def read_files():
    #print(pd.__version__)
    #Only use trainning dataset, since testing dataset has no label to evaluate
    data_dir = os.listdir("./train/train")
    col_names=["t","centre_2d_x","centre_2d_y","bb_2d_br_x","bb_2d_br_y","bb_2d_tl_x","bb_2d_tl_y","centre_3d_x","centre_3d_y","centre_3d_z","bb_3d_brb_x","bb_3d_brb_y","bb_3d_brb_z","bb_3d_flt_x","bb_3d_flt_y","bb_3d_flt_z"]
    res = pd.DataFrame()
    #First build a map(bucket array) to map timestamp(int) to activities
    for sub_dir in data_dir:
        #print(sub_dir)
        df = pd.DataFrame()
        bucket = [] #use a bucket array to save labels at each timestamp(roughly)
        init = False # haven't initiate the bucket 
        folder = os.listdir("./train/train/"+sub_dir)
        for file in folder:
            if "annotations_0" in file:   
                file_name = "./train/train/"+sub_dir+"/"+file
                with open(file_name,'r') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader) #skip first row
                    for row in reversed(list(reader)):#Read from the end of the file, so that the bucket array can be initialized at the beginning
                        start = math.ceil(float(row[0])*10) # *10 to get a finer map
                        end = math.ceil(float(row[1])*10)
                        label = int(row[3])
                        if init is False:#
                            bucket = [-1 for x in range(end)] # -1 means no label for this timestamp
                            init = True
                        for i in range(start,end):
                            bucket[i] = label               
        #Secondly combine the video files add activity labels for each row and delete those with no activity labels
        for file in folder:  
            #print(file)
            if "video" in file:
                file_name = "./train/train/"+sub_dir+"/"+file
                procData = pd.read_table(file_name, sep=',') 
                procData.columns = col_names
                procData['label'] = procData["t"].apply(lambda x: -1 if x*10 >= len(bucket) else bucket[math.floor(x*10)])
                # add the loc column for the videos location
                if "hallway" in file:
                    procData['loc'] = 1
                elif "kitchen" in file:
                    procData['loc'] = 2
                else:
                    procData['loc'] = 3
                    
                df = df.append(procData, ignore_index=True)
                
                df = df[df.label!=-1] #drop all rows with label = -1
                df.sort_values(by=['t'], inplace=True)
        
        #Last, add acceleration data
        for file in folder:
            if "acceleration" in file:
                file_name = "./train/train/"+sub_dir+"/"+file
                accData = pd.read_table(file_name, sep=',') 
                accData.fillna(0,inplace=True)
                accData[['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']] = accData[['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']].apply(lambda x: x*(-1)) # convert the negative signal to positive. Not necessary. 
                df = pd.merge_asof(left=df,right=accData,on='t',tolerance=1)
                df.dropna(subset=['label'],inplace=True) #remove the rows with no labels
                df.dropna(subset=['x','y','z'],inplace=True)
        df['folder_id'] = int(sub_dir)
        res = res.append(df, ignore_index=True)
   
    res['activities'] = res['label']#move the label to the last column and change the column name to activities
    res = res.drop(['t','label'],axis=1)#remove timestamp and label
    #res = res.drop(['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP'],axis=1)#remove timestamp and label
    
    #print(set(res['activities']))
    res = scale(res)
    
    #res.to_csv('prepocessed_1.csv')
    return res

def scale(df):#pandas dataframe
    col_names = ["centre_2d_x","centre_2d_y","bb_2d_br_x","bb_2d_br_y","bb_2d_tl_x","bb_2d_tl_y","centre_3d_x","centre_3d_y","centre_3d_z","bb_3d_brb_x","bb_3d_brb_y","bb_3d_brb_z","bb_3d_flt_x","bb_3d_flt_y","bb_3d_flt_z",'x','y','z','Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    df[col_names] = scaler.fit_transform(df[col_names])
    return df
    
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
            y.append(data[start][-1]) 
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][-1] != data[start+1][-1]:
                    break
                start += 1
            start += 1
    #print(np.asarray(y))
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}

def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.')  
    
if __name__ == "__main__":
    file_name = 'sphere.h5'
    window_size = 20
    data = read_files()
    numpy_data = data.to_numpy()
    #numpy_data = downsize(numpy_data) 
    segment_data = segment(numpy_data, window_size) 
    save_data(segment_data, file_name)