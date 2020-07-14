# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:09:38 2020

@author: Jieyun Hu
"""

# This file is for PAMAP2 data processing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
activities = {1: 'stand',
              2: 'walk',
              4: 'sit',
              5: 'lie',
              101: 'relaxing',
              102: 'coffee time',
              103: 'early morning',
              104: 'cleanup',
              105: 'sandwich time'
               }

def read_files():
    #pick partial data from dataset
    list_of_files = ['./dataset/S1-ADL1.dat',
                     './dataset/S1-ADL2.dat',
                     './dataset/S1-ADL3.dat',
                     './dataset/S1-ADL4.dat',
                     './dataset/S2-ADL1.dat',
                     './dataset/S2-ADL2.dat',
                     './dataset/S2-ADL3.dat',
                     './dataset/S2-ADL4.dat',
                     './dataset/S3-ADL1.dat',
                     './dataset/S3-ADL2.dat',
                     './dataset/S3-ADL3.dat',
                     './dataset/S3-ADL4.dat',
                     './dataset/S4-ADL1.dat',
                     './dataset/S4-ADL2.dat',
                     './dataset/S4-ADL3.dat',
                     './dataset/S4-ADL4.dat',                 
                     ]
    
    list_of_drill = ['./dataset/S1-Drill.dat',
                     './dataset/S2-Drill.dat',
                     './dataset/S3-Drill.dat',
                     './dataset/S4-Drill.dat',
                     ]
    col_names = []
    with open('col_names','r') as f:# a file with all column names was created
        lines = f.read().splitlines()
        for line in lines:
            col_names.append(line)
    print(len(col_names))
       
    dataCollection = pd.DataFrame()
    for i, file in enumerate(list_of_files):
        print(file," is reading...")
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = col_names
        procData['file_index'] = i # put the file index at the end of the row
        dataCollection = dataCollection.append(procData, ignore_index=True)       
        #break; # for testing short version, need to delete later       
    dataCollection.reset_index(drop=True, inplace=True)
    
    return dataCollection


def dataCleaning(dataCollection):
    dataCollection = dataCollection.loc[:,dataCollection.isnull().mean()< 0.1] #drop the columns which has NaN over 10%
    #print(list(dataCollection.columns.values))
    dataCollection = dataCollection.drop(['MILLISEC', 'LL_Left_Arm','LL_Left_Arm_Object','LL_Right_Arm','LL_Right_Arm_Object', 'ML_Both_Arms'],
                                        axis = 1)  # removal of columns not related, may include others.
    
    dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
    
    print(dataCollection.isna().sum().sum())#count all NaN 
    print(dataCollection.shape)
    #dataCollection = dataCollection.dropna()
    dataCollection = dataCollection.interpolate() 
    print(dataCollection.isna().sum().sum())#count all NaN 
    #removal of any remaining NaN value cells by constructing new data points in known set of data points
    #for i in range(0,4):
    #    dataCollection["heartrate"].iloc[i]=100 # only 4 cells are Nan value, change them manually
    print("data cleaned!")
    return dataCollection

def reset_label(dataCollection, locomotion): 
    # Convert original labels {1, 2, 4, 5, 101, 102, 103, 104, 105} to new labels. 
    mapping = {1:1, 2:2, 5:0, 4:3, 101: 0, 102:1, 103:2, 104:3, 105:4} # old activity id to new activity Id 
    if locomotion: #new labels [0,1,2,3]
        for i in [5,4]: # reset ids in Locomotion column
            dataCollection.loc[dataCollection.Locomotion == i, 'Locomotion'] = mapping[i]
    else: # reset the high level activities ; new labels [0,1,2,3,4]
        for j in [101,102,103,104,105]:# reset ids in HL_activity column
            dataCollection.loc[dataCollection.HL_Activity == j, 'HL_Activity'] = mapping[j]
    return dataCollection

def segment_locomotion(dataCollection, window_size): # segment the data and create a dataset with locomotion classes as labels
    #remove locomotions with 0
    dataCollection = dataCollection.drop(dataCollection[dataCollection.Locomotion == 0].index)
    # reset labels
    dataCollection= reset_label(dataCollection,True)
    #print(dataCollection.columns)
    loco_i = dataCollection.columns.get_loc("Locomotion")
    #convert the data frame to numpy array
    data = dataCollection.to_numpy()
    #segment the data
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][loco_i] == data[end][loco_i] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the file
            X.append(data[start:(end+1),0:loco_i])
            y.append(data[start][loco_i])
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][loco_i] != data[start+1][loco_i]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}

def segment_high_level(dataCollection, window_size): # segment the data and create a dataset with high level activities as labels
    #remove locomotions with 0
    dataCollection = dataCollection.drop(dataCollection[dataCollection.HL_Activity == 0].index)
    # reset labels
    dataCollection= reset_label(dataCollection,False)
    #print(dataCollection.columns)
    HL_Activity_i = dataCollection.columns.get_loc("HL_Activity")
    #convert the data frame to numpy array
    data = dataCollection.to_numpy()
    #segment the data
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][HL_Activity_i] == data[end][HL_Activity_i] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the file
            #print(data[start:(end+1),0:(HL_Activity_i)])
            X.append(data[start:(end+1),0:(HL_Activity_i-1)])# slice before locomotion
            y.append(data[start][HL_Activity_i])
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][HL_Activity_i] != data[start+1][HL_Activity_i]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}

def plot_series(df, colname, act, file_index, start, end):
    unit='ms^-2'
    #pylim =(-25,25)
    #print(df.head())
    print(set(df.loc[df.file_index == file_index, "Locomotion"]))
    df1 = df[(df.Locomotion == act) & (df.file_index == file_index)]
    #df1 = df[(df.HL_Activity ==act) & (df.file_index == file_index)]
    if df1.shape[0] < 1:
        print("Didn't find the region. Please reset activityID and subject_id")
        return
    df_len = df1.shape[0]
    if df_len > start and df_len  > end:
        df1 = df1[start:end]
    elif df_len  > start and df_len  <= end:
        df1 = df1[start:df_len]
    else:
        print("Out of boundary, please reset the start and end points")
        return
    print(df1.shape)
    #print(df1.head(10))
    plottitle = colname +' - ' + str(act)
    #plotx = colname
    fig = df1[colname].plot()
    #print(df.index)
    #ax1 = df1.plot(x=df.index,y=plotx, color='r', figsize=(12,5), ylim=pylim)
    fig.set_title(plottitle)
    fig.set_xlabel('window')
    fig.set_ylabel(unit)
    

def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.')    

if __name__ == "__main__":   
    window_size = 25   
    df = read_files()
    df = dataCleaning(df)
    #plot_series(df, colname, act, file_index, start, end)
    #plot_series(df, "Acc-RKN^-accX", 4, 2, 100, 150)
    
    loco_filename = "loco_2.h5" # "loco.h5" is to save locomotion dataset. 
    data_loco = segment_locomotion(df, window_size)
    save_data(data_loco,loco_filename)
    
    hl_filename = "hl_2.h5" #"hl.h5" is to save high level dataset
    data_hl = segment_high_level(df, window_size)
    save_data(data_hl,hl_filename)
    
    
    
    
    
    
    

    
    
