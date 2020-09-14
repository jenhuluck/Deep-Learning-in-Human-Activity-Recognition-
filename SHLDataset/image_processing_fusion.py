# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:25:27 2020

@author: Jieyun Hu
"""

#image processing for data fusion
import numpy as np
import h5py
import math
from keras.preprocessing.image import load_img, img_to_array
from collections import Counter
 
# Four motion.txt are all with the same timestamp, so no need to synchonize them.
# motion.txt are not synchonized with images, so first need to map frame No. to each timestamp in motion data
# Based on document, ts = offset1 + speedup * tv
# fps = 0.5, 1 frame in every two seconds, so frame0 = 0ms, frame1 = 2000ms, frame2 = 4000ms... framen = 2000*n ms
# ts = offset1 + speedup * frameNo * 2000 => frameNo = (ts - offset1)/speedup/2000
def add_frame():
    
    bag_motion = ["./SHLDataset_preview_v1/User1/220617/Bag_Motion.txt","./SHLDataset_preview_v1/User1/260617/Bag_Motion.txt","./SHLDataset_preview_v1/User1/270617/Bag_Motion.txt"]
    labels = ["./SHLDataset_preview_v1/User1/220617/Label.txt","./SHLDataset_preview_v1/User1/260617/Label.txt","./SHLDataset_preview_v1/User1/270617/Label.txt"]
    offset_paths = ["./SHLDataset_preview_v1/User1/220617/videooffset.txt","./SHLDataset_preview_v1/User1/260617/videooffset.txt","./SHLDataset_preview_v1/User1/270617/videooffset.txt"]
    speedup_paths = ["./SHLDataset_preview_v1/User1/220617/videospeedup.txt","./SHLDataset_preview_v1/User1/260617/videospeedup.txt","./SHLDataset_preview_v1/User1/270617/videospeedup.txt"]   
    
    data = np.array([])
    for i, file in enumerate(bag_motion):
        np_motion = np.loadtxt(file, usecols = [0]).reshape(-1,1)
        np_motion = downsize(np_motion).astype(np.int64)
        
        # attach frame No
        # get offset1 and speedup value
        offset_path = offset_paths[i]
        speedup_path = speedup_paths[i]
        offset1 = get_offset1(offset_path)
        speedup = get_speedup(speedup_path)        
        frame_No = np.apply_along_axis(lambda x: int((x[0] - offset1)/speedup/2000), 1, np_motion).reshape(-1,1)
 
        # attach label
        start = np_motion[0,0].astype(np.int64)
        end = np_motion[-1,0].astype(np.int64)
        label = labels[i]
        np_label = np.loadtxt(label)        
        start_index = np.where(np_label == start)[0][0] # find the row index of start timestamp 
        end_index = np.where(np_label == end)[0][0] # find the column index of start timestamp 
        np_label = find_labels(np_label,start_index,end_index)
        
        folder_id = np.full(np_label.shape, i+1) # put folder_id in the last column
        
        concatenate = np.concatenate((np_motion,frame_No, np_label,folder_id),axis=1)
        if i == 0:
            data = concatenate
        else:
            data = np.concatenate((data,concatenate))
        #break; #testing
    print(data.shape)
    return data

#parse the videooffset.txt
def get_offset1(file):
    f = open(file,'r')
    line = f.readline()
    p = line.split(" ")
    offset1 = int(float(p[0].rstrip("\n")))
    f.close()
    return offset1

#parse the videospeedup.txt
def get_speedup(file):  
    f = open(file,'r')
    speedup = int(f.readline())    
    f.close()
    return speedup

    
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

# The same algorithm as motion data segmentation so that the images map the same labels with motion data
def segment(data, window_size): # data is numpy array
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][-2]!=0 and data[start][-2] == data[end][-2] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the same object
            X.append(data[start:(end+1),[1,-1]])
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
    #print(np.asarray(X).shape)
    return (X, y)

# load the images based on segmentation
# feed in segmented data
# i.e. (X,y)
# get the frame_no from the first entry in the segment
# Other strategy: find the most frequent frame_no in the segment
def load_images(data):
    X = data[0]
    y = data[1]
    images = []
    for seg in X:
        #frame_num = int(seg[0][0])
        frame_num = int(Counter(seg[:,0]).most_common(1)[0][0])
        folder_id = int(seg[0][1])
        image_path = "./picture{}/frame{}.jpg".format(folder_id,frame_num)
        img = load_img(image_path,color_mode='grayscale',target_size=(100,100)) # make size smaller to save memory
        img = img_to_array(img).astype('float32')/255
        img = img.reshape(img.shape[0],img.shape[1])
        images.append(img)
    images = np.asarray(images)
    y = np.asarray(y, dtype=int)
    print(images.shape)
    print(y.shape)
    return {'inputs' : images, 'labels': y}
  
    #print(np.asarray(X).shape, np.asarray(y).shape)
    #return {'inputs' : np.asarray(X), 'labels': np.asarray(y,dtype=int)}
if __name__ == "__main__":
    file_name = "image_for_fusion.h5"
    data = add_frame()
    seg = segment(data,20)
    img = load_images(seg)  
    #data = prepare_dataset()
    save_data(img,file_name)