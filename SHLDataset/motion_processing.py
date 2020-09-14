# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:46:54 2020

@author: Jieyun Hu
"""

# motion proprocessing
#1. downsize the data
#2. find label for each timestamp
#3. remove null lables
#4. slice windows
import numpy as np
import h5py
# For each type of motion, read *_Motion.txt and Label.txt together
# Find the start label and end label line which map to the start and end timestamp of the motion table, truncate it and concatanate it to the end of motion data
def read_motion(chose):
    motions =[]
    bag_motion = ["./SHLDataset_preview_v1/User1/220617/Bag_Motion.txt","./SHLDataset_preview_v1/User1/260617/Bag_Motion.txt","./SHLDataset_preview_v1/User1/270617/Bag_Motion.txt"]
    hand_motion = ["./SHLDataset_preview_v1/User1/220617/Hand_Motion.txt","./SHLDataset_preview_v1/User1/260617/Hand_Motion.txt","./SHLDataset_preview_v1/User1/270617/Hand_Motion.txt"]
    hip_motion = ["./SHLDataset_preview_v1/User1/220617/Hips_Motion.txt","./SHLDataset_preview_v1/User1/260617/Hips_Motion.txt","./SHLDataset_preview_v1/User1/270617/Hips_Motion.txt"]
    torso_motion = ["./SHLDataset_preview_v1/User1/220617/Torso_Motion.txt","./SHLDataset_preview_v1/User1/260617/Torso_Motion.txt","./SHLDataset_preview_v1/User1/270617/Torso_Motion.txt"]
    labels = ["./SHLDataset_preview_v1/User1/220617/Label.txt","./SHLDataset_preview_v1/User1/260617/Label.txt","./SHLDataset_preview_v1/User1/270617/Label.txt"]
    
    if chose == 1:
        motions = bag_motion
    elif chose == 2:
        motions = hand_motion
    elif chose == 3:
        motions = hip_motion
    else:
        motions = torso_motion
    
    data = np.array([]) 
    for i, file in enumerate(motions):
        np_motion = np.loadtxt(file)
        np_motion = downsize(np_motion)
        print(np_motion.shape)
        start = np_motion[0,0].astype(np.int64)
        end = np_motion[-1,0].astype(np.int64)
        label = labels[i]
        np_label = np.loadtxt(label)
        start_index = np.where(np_label == start)[0][0] # find the row index of start timestamp 
        end_index = np.where(np_label == end)[0][0] # find the column index of start timestamp 
        np_label = find_labels(np_label,start_index,end_index)
        folder_id = np.full(np_label.shape, i) # put folder_id in the last column
        concatenate = np.concatenate((np_motion,np_label,folder_id),axis=1)
        if i == 0:
            data = concatenate
        else:
            data = np.concatenate((data,concatenate))
        #break; #testing
    print(data.shape)
    return data
        
    '''
    test = "./SHLDataset_preview_v1/User1/220617/Hand_Motion.txt"    
    np_motion = np.loadtxt(test)
    np_motion = downsize(np_motion)
    print(np_motion.shape)
    start = np_motion[0,0].astype(np.int64)
    end = np_motion[-1,0].astype(np.int64)
    
    print(start)
    print(end)
    
    label = "./SHLDataset_preview_v1/User1/220617/Label.txt"
    np_label = np.loadtxt(label)
    start_index = np.where(np_label == start)[0][0] # find the row index of start timestamp 
    end_index = np.where(np_label == end)[0][0] # find the column index of start timestamp 
    np_label = find_labels(np_label,start_index,end_index)
    print(np_label.shape)
    
    concate = np.concatenate((np_motion,np_label),axis=1)
    print(concate.shape)

    '''
def segment(data, window_size): # data is numpy array
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][-2]!=0 and data[start][-2] == data[end][-2] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the same object
            X.append(data[start:(end+1),1:-2])
            y_label = data[start][-2]
            if y_label == 8:
                y.append(0) # change label 8 to 0
            else:
                y.append(data[start][-2])
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][-2] == 0 or data[start][-2] != data[start+1][-2]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}


def find_labels(labels,start_index, end_index):
    interval = 10
    label_col_index = 1 # the 2 column of Label.txt
    data = labels[start_index: end_index+1:interval, label_col_index].reshape(-1,1) # need to be the shape like (n,1), so that it can be concatenate later
    return data

def downsize(data):# data is numpy array
    downsample_size = 10
    data = data[::downsample_size,:]
    return data
    
def save_data(data,file_name): # save the data in h5 format
    f = h5py.File(file_name,'w')
    for key in data:
        print(key)
        f.create_dataset(key,data = data[key])       
    f.close()
    print('Done.')  
    
    
if __name__ == "__main__":
    # The sensors are located in different body part. Choose a body part to preprocess that sensor data
    # 1 : bag ; 2: hand; 3:hip; 4: torso
    chose = 4
    file = "" 
    if chose == 1:
        file = "bag.h5"
    elif chose == 2:
        file = "hand.h5"
    elif chose == 3:
        file = "hip.h5"
    else:
        file = "torso.h5"
    
    data = read_motion(chose)
    segment_data = segment(data,20)
    #save_data(segment_data, file)
    