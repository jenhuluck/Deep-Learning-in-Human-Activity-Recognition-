## Dataset

The University of Sussex-Huawei Locomotion (SHL) dataset was used in our research. It is a versatile annotated dataset of modes of locomotion and transportation of mobile users. The dataset can be downloaded from http://www.shl-dataset.org/download/. The dataset was recorded over 7 months by 3 users engaging in 8 different transportation modes: still, walk, run, bike, car, bus, train and subway. The dataset contains multi-modal data from a body-worn camera and from 4 smartphones, carried simultaneously at four body locations.


## Methods and Results
1.	Proposed data processing methods involve video conversion, data labeling, data segmentation.
2.	Experimented using three deep learning methods (FFNN, CNN, RNN) in HAR datasets. All the models worked well and achieved a high accuracy using the dataset. Using CNN models achieved slightly higher accuracy than the other two.
3.	Two data fusion methods including early and late fusion were compared in our study using multimodal data collected by smartphone. The early fusion performed slightly better than the late fusion.
4.	A CNN based model was trained on both motion data from smartphone and image data derived from videos. The performance was greatly improved (99.92%) after 50 epochs of training. The model is also stable and less sensitive to noises. Moreover, we tested on fusing only one motion sensor data with image data, which also achieved a very high accuracy (99.97%) indicating fusing these two sensor data is sufficient to achieve a good performance using this dataset.

## Data proprocessing
![alt text](https://github.com/jenhuluck/deep-learning-in-ADL/blob/master/SHLDataset/figures/dataprocessing.png?raw=true)

## Data fusion model 
![alt text](https://github.com/jenhuluck/deep-learning-in-ADL/blob/master/SHLDataset/figures/fusion_model.png?raw=true)

## Note
I am still working on the project as the research project of my master degree. Please do not use the figures and results from this research. 

  
