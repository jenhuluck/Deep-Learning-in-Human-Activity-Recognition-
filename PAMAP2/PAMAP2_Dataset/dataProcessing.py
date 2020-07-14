# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:09:38 2020

"""

# This file is for PAMAP2 data processing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
activityIDdict = {0: 'transient',
              1: 'lying',# no change in index
              2: 'sitting',# no change in index
              3: 'standing',# no change in index
              4: 'walking',# no change in index
              5: 'running',# no change in index
              6: 'cycling',# no change in index
              7: 'Nordic_walking',# no change in index
              9: 'watching_TV', # not in dataset
              10: 'computer_work',# not in dataset
              11: 'car driving', # not in dataset
              12: 'ascending_stairs', # new index:8
              13: 'descending_stairs', # new index:9
              16: 'vacuum_cleaning', # new index:10
              17: 'ironing', # new index:11
              18: 'folding_laundry',# not in dataset
              19: 'house_cleaning', # not in dataset
              20: 'playing_soccer', # not in dataset
              24: 'rope_jumping' # new index: 0 
              }
#{24:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,12:8,13:9,16:10,17:11}
              
def read_files():
    list_of_files = ['./Protocol/subject101.dat',
                     './Protocol/subject102.dat',
                     './Protocol/subject103.dat',
                     './Protocol/subject104.dat',
                     './Protocol/subject105.dat',
                     './Protocol/subject106.dat',
                     './Protocol/subject107.dat',
                     './Protocol/subject108.dat',
                     './Protocol/subject109.dat' ]
    
    subjectID = [1,2,3,4,5,6,7,8,9]
    

    
    colNames = ["timestamp", "activityID","heartrate"]
    
    IMUhand = ['handTemperature', 
               'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
               'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
               'handGyro1', 'handGyro2', 'handGyro3', 
               'handMagne1', 'handMagne2', 'handMagne3',
               'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']
    
    IMUchest = ['chestTemperature', 
               'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
               'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
               'chestGyro1', 'chestGyro2', 'chestGyro3', 
               'chestMagne1', 'chestMagne2', 'chestMagne3',
               'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']
    
    
    IMUankle = ['ankleTemperature', 
               'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
               'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
               'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
               'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
               'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']
    
    columns = colNames + IMUhand + IMUchest + IMUankle
    
    dataCollection = pd.DataFrame()
    for file in list_of_files:
        print(file," is reading...")
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = columns
        procData['subject_id'] = int(file[-5])
        dataCollection = dataCollection.append(procData, ignore_index=True)
        
        #break; # for testing short version, need to delete later
        
    dataCollection.reset_index(drop=True, inplace=True)
    
    return dataCollection


def dataCleaning(dataCollection):
    dataCollection = dataCollection.drop(['timestamp', 'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
                                         'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
                                         'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
                                         axis = 1)  # removal of orientation columns as they are not needed
    dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
    dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
    dataCollection = dataCollection.dropna()
    #dataCollection = dataCollection.interpolate() 
    #removal of any remaining NaN value cells by constructing new data points in known set of data points
    #for i in range(0,4):
    #    dataCollection["heartrate"].iloc[i]=100 # only 4 cells are Nan value, change them manually
    print("data cleaned!")
    return dataCollection

def reset_label(dataCollection): 
    # Convert original labels {1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24} to new labels. 
    mapping = {24:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,12:8,13:9,16:10,17:11} # old activity Id to new activity Id 
    for i in [24,12,13,16,17]:
        dataCollection.loc[dataCollection.activityID == i, 'activityID'] = mapping[i]

    return dataCollection

def segment(data, window_size): # data is numpy array
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][0] == data[end][0] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the same object
            X.append(data[start:(end+1),1:-1])
            y.append(data[start][0])
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][0] != data[start+1][0]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}

def downsize(data):# data is numpy array
    downsample_size = 3
    data = data[::downsample_size,:]
    return data

def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.')    


def plot_series(df, colname, act, subject, start, end):
    unit='ms^-2'
    #pylim =(-25,25)
    #print(df.head())
    df1 = df[(df.activityID ==act) & (df.subject_id == subject)]
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
    #fig.show()
    
#visualize the curve in a given window
#with same subject and same activity
#feat:[]
#def visualize(act_id,)
    


if __name__ == "__main__":
    file_name = 'pamap1.h5'
    window_size = 25
    data = read_files()
    data = dataCleaning(data)
    #plot_series(data,'handAcc16_1',1,1,400,500)
    #plot_series(data,'chestAcc16_1',1,1,400,500)
    #plot_series(data,'ankleAcc16_1',1,1,400,500)
    data = reset_label(data)   
    numpy_data = data.to_numpy()
    numpy_data = downsize(numpy_data) # downsize to 30%
    
    segment_data = segment(numpy_data, window_size)   
    #save_data(segment_data, file_name)
    

    
    
