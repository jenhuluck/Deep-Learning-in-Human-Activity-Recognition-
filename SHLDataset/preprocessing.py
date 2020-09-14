# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:31:33 2020

@author: Jieyun Hu
"""
#1. transfer the videos to grayscale images and rescale them
#2. find the label for each frame and save the labels data in the same file
import cv2
import os

#read videos and convert each frame into an image
#grayscale and resize the image
#save the images
def read_videos():
    video_path = ["./SHLDataset_preview_v1/User1/220617/timelapse.avi","./SHLDataset_preview_v1/User1/260617/timelapse.avi","./SHLDataset_preview_v1/User1/270617/timelapse.avi"]
    folder_count = 1
    for path in video_path: 
        save_path = "./picture%d"%folder_count        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        interval = 1/fps * 1000 # For example, if fps = 0.5, then it is one frame in every 2 senconds, so interval time of two frame is 2000 milliseconds
        #print("video FPS {}".format(vidcap.get(cv2.CAP_PROP_FPS)))
        # read frames and save to images at fps_save
        success,image = vidcap.read()
        count = 0   
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*interval)) 
            success,image = vidcap.read()
            if success:
                img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #grayscale
                img_resized =cv2.resize(img_gray,(100,100)) #resize
                print ('Read a new frame: ', success)
                cv2.imwrite( save_path + "\\frame%d.jpg" % count, img_resized)     # save frame as JPEG file
                count = count + 1
        frame_num = open(save_path+"\\frame_number.txt","w")
        frame_num.write(str(count)) # save the total frame number
        frame_num.close()
        folder_count += 1
        
#read label.txt and save it as a map
#line number as key, label as value
def line_map(file):
    #file = "./SHLDataset_preview_v1/User1/220617/Label.txt"
    f = open(file,'r')
    lines = f.readlines()
    line_num = 1
    line_map = dict()
    for line in lines:
        p = line.split(" ")
        label = int(p[1]) # use the 2nd column as label
        line_map[line_num] = label
        line_num += 1
    return line_map    

# from the document l = offset + speedup*tv/10
def cal_line(offset,speedup,frame_num):
    # fps = 0.5, one frame in every 2 seconds
    interval = 2000
    tv = frame_num * interval #start from frame 0
    return int(offset+ speedup*tv/10)
 
#create a label.txt file in each picture folder 
#label.txt saves the label for each frame
def save_labels():
    label_paths = ["./SHLDataset_preview_v1/User1/220617/label.txt","./SHLDataset_preview_v1/User1/260617/label.txt","./SHLDataset_preview_v1/User1/270617/label.txt"]
    picture_paths = ["./picture1","./picture2","./picture3"]
    offset_paths = ["./SHLDataset_preview_v1/User1/220617/videooffset.txt","./SHLDataset_preview_v1/User1/260617/videooffset.txt","./SHLDataset_preview_v1/User1/270617/videooffset.txt"]
    speedup_paths = ["./SHLDataset_preview_v1/User1/220617/videospeedup.txt","./SHLDataset_preview_v1/User1/260617/videospeedup.txt","./SHLDataset_preview_v1/User1/270617/videospeedup.txt"]   
    i = 0
    for path in label_paths:       
        pic_path = picture_paths[i]
        offset_path = offset_paths[i]
        speedup_path = speedup_paths[i]
        
        offset = get_offset2(offset_path)
        speedup = get_speedup(speedup_path)
        
        frame_file = open(pic_path+"\\frame_number.txt","r")
        frame_num = int(frame_file.readline()) 
        label_map = line_map(path) 
        
        f = open(pic_path + "\\labels.txt","w")
        for frame in range(frame_num):
            line_num = cal_line(offset,speedup,frame)
            label = label_map[line_num]            
            f.write(str(label)+"\n") 
        f.close()
        frame_file.close()
        i += 1
        
#parse the videooffset.txt
def get_offset2(file):
    f = open(file,'r')
    line = f.readline()
    p = line.split(" ")
    offset2 = int(float(p[1].rstrip("\n")))
    f.close()
    return offset2

#parse the videospeedup.txt
def get_speedup(file):  
    f = open(file,'r')
    speedup = int(f.readline())    
    f.close()
    return speedup

if __name__ == '__main__':
    read_videos()
    save_labels()

    

    
    
    
