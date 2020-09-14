# SPHERE Challenge: Activity Recognition with Multimodal Sensor Data 

All use of the data must cite the following: 

> Niall Twomey, Tom Diethe, Meelis Kull, Hao Song, Massimo Camplani, Sion Hannuna, Xenofon Fafoutis, Ni Zhu, Pete Woznowski, Peter Flach, and Ian Craddock. _The SPHERE Challenge: Activity Recognition with Multimodal Sensor Data_. 2016.

This dataset has an associated homepage: 
* http://irc-sphere.ac.uk/sphere-challenge/home

The following pages are available in the dataset website:

* Prizes and Timelines: http://irc-sphere.ac.uk/sphere-challenge/prizes
* Task Statement: http://irc-sphere.ac.uk/sphere-challenge/task
* Annotations: http://irc-sphere.ac.uk/sphere-challenge/annotation
* Sensor Description: http://irc-sphere.ac.uk/sphere-challenge/sensors
* Data Formats: http://irc-sphere.ac.uk/sphere-challenge/data
* Evaluation Metrics: http://irc-sphere.ac.uk/sphere-challenge/evaluation
* Challenge Rules: http://irc-sphere.ac.uk/sphere-challenge/rules

A number of processing and visualisation scripts  can be found in the challenge Github repository: 

* https://github.com/IRC-SPHERE/sphere-challenge

## INTRODUCTION

The task of this challenge is to predict aspects the activities of residents within a smart home based only on observed sensor data. Sensor data are obtained from the following three sensing modalities:

* A wrist-worn accelerometer
* Video + Depth (RGB-D)
* Passive environmental presence sensors

The data and file formats are described in the following section. 

The train and test subsets can be found in the train.zip and test.zip files, and metadata.zip contains meta data regarding the sensor names, room names, activity names. supplementary.zip contains floor-plans of the house, images from the three cameras, and the script used for the training data collection

## DATA

Training data and testing data can be found in the ‘train’ and ‘test’ subdirectories respectively. The recorded data are collected under unique codes (each recording will be referred to as a ‘data sequence’). Timestamps are rebased to be relative to the start of the sequences, i.e. for a sequence of length 10 seconds, all timestamps will be within the range 0-10 seconds. 

Each data sequence contains the following files:

* targets.csv (available only with training data)
* pir.csv
* video_hallway.csv
* video_living_room.csv
* video_kitchen.csv
* meta.json

The following files are also available within the training directory:

* annotations_*.csv
* locations_*.csv

The data from annotations_*.csv is used to create the targets.csv file, and locations_\*.csv files are available for participants that want to model indoor localisation. These are only available for the training set.

The dataset may be downloaded from data.bris: 

* https://data.bris.ac.uk/data/

### targets.csv (available in train only)

This file contains the probabilistic targets for classification. Multiple annotators may have annotated each sequence, and this file aggregates all of the annotations over one second windows. The mean duration of each label within this window is used as the target variable. 

The following 20 activities are labelled:

`annotation_names = ('a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk', 'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie', 't_sit_stand', 't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn')`

The prefix ‘a_’ indicates an ambulation activity (i.e. an activity consisting of continuing movement), ‘p_’ annotations indicate static postures (i.e. times when the participants are stationary), and ‘t_’ annotations indicate posture-to-posture transitions.

This file contains of 22 columns:

* start: The starting time of the window
* end: The ending time of the window
* targets: Columns 3-22: the 20 probabilistic targets.



### pir.csv (available for train and test)

This file contains the start time and duration for all PIR sensors in the smart environment. A PIR sensor is located in every room:

`pir_locations = ('bath', 'bed1', 'bed2', 'hall', 'kitchen', 'living', 'stairs', 'study', 'toilet')`

The columns of this CSV file are:

* start: the start time of the PIR sensor (relative to the start of the sequence)
* end: the end time of the PIR sensor (relative to the start of the sequence)
* name: the name of the PIR sensor being activated (from the above list)
* index: the index of the activated sensor from the pir_locations list starting at 0



### acceleration.csv (available for train and test)

The acceleration file consists of eight columns:

* t: this is the time of the recording (relative to the start of the sequence)
* x/y/z: these are the acceleration values recorded on the x/y/z axes of the accelerometer.
* Kitchen_AP/Lounge_AP/Upstairs_AP/Study_AP: these specify the received signal strength (RSSI) of the acceleration signal as received by the access kitchen/lounge/upstairs access points. Empty values indicate that the access point did not receive the packet.



### video_*.csv (available for train and test)

The following columns are found in the video_hallway.csv, video_kitchen.csv and video_living_room.csv files:

* t: The current time (relative to the start of the sequence)
* centre_2d_x/centre_2d_y: The x- and y-coordinates of the centre of the 2D bounding box.
* bb_2d_br_x/bb_2d_br_y: The x and y coordinates of the bottom right (br) corner of the 2D bounding box
* bb_2d_tl_x/bb_2d_tl_y: The x and y coordinates of the top left (tl) corner of the 2D bounding box
* centre_3d_x/centre_3d_y/centre_3d_z: the x, y and z coordinates for the centre of the 3D bounding box
* bb_3d_brb_x/bb_3d_brb_y/bb_3d_brb_z: the x, y, and z coordinates for the bottom right back corner of the 3D bounding box
* bb_3d_flt_x/bb_3d_flt_y/bb_3d_flt_z: the x, y, and z coordinates of the front left top corner of the 3D bounding box.


### meta.json
This file contains meta-data regarding the duration of the sequence, and the unique annotator codes. These are in JSON format (see http://www.json.org/) and can be viewed in any popular text editor.


## SUPPLEMENTARY FILES

The following two sets of file need not be used for the challenge, but are included to facilitate users that wish to perform additional modelling of the sensor environment.


### locations_*.csv (available in train only)

This labels the room that is currently occupied by the recruited participant. The following rooms are labelled:

`location_names = ('bath', 'bed1', 'bed2', 'hall', 'kitchen', 'living', 'stairs', 'study', 'toilet')`

locations.csv contains the following four columns:

* start: the time a participant entered a room (relative to the start of the sequence)
* end: the time the participant left the room (relative to the start of the sequence)
* name: the name of the room (from the above list)
* index: the index of the room name starting at 0


### annotations_*.csv (available in train only)

annotations.csv contains the annotations that were provided by the annotators. Each file contains the following:

* start: the start time of the activity (relative to the start of the sequence)
* end: the end time of the activity (relative to the start of the sequence)
* name: the name of the label (from the list of `annotation_names`)
* index: the index of the label name starting at 0


## METADATA

metadata.zip contains meta data regarding the sensor names, room names, activity names, all in JavaScript Object Notation (JSON) format. This contains the following files:

### accelerometer_axes.json
This file contains the order of the accelerometer axes (i.e. x, y, z)

### access_point_names.json
This file contains the name of the access points in the house.

### annotations.json
This file contains all of the activity labels.

### pir_locations.json
This file contains the locations of the PIR sensors.

### rooms.json
This file contains the names of all of the rooms in the house.

### video_feature_names.json
This file contains the feature names of the video sensors, and their dimensionalities.

### video_locations.json
This file contains the locations of the video sensors.

supplementary.zip contains floor-plans of the house, images from the three cameras, and the script used for the training data collection. This contains the following files:

### data_collection_script.pdf contains the script used for collecting the data that is used for the training data and some of the testing data, as described in the paper.

### floorpan.pdf contains the floor-plans of the two floors of the house, with the approximate locations of the cameras shown on the ground floor image. 

### hallway.png is a still image from the camera in the hallway.

### kitchen.png is a still image from the camera in the kitchen.

### living_room.png is a still image from the camera in the living room.

