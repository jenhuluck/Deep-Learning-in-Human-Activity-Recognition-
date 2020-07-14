## Introduction
This repository is to apply deep learning models on Human Activity Recognition(HAR)/Activities of Daily Living(ADL) datasets. Three deep learning models, including Convolutional Neural Networks(CNN), Deep Feed Forward Neural Networks(DNN) and Recurrent Neural Networks(RNN) were applied to the datasets. Four HAR/ADL benchmark datasets were tested. The goal is to gain some experiences on handling the sensor data as well as classifying human activities using deep learning. 

## Benchmark datasets
  * [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heat rate monitor.
  * [OPPORTUNITY dataset](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) contains data of 35 ADL activities (13 low-level, 17 mid-level and 5 high-level) which were collected through 23 body worn sensors, 12 object sensors, 21 ambient sensors. 
  * [Daphnet Gait dataset](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait) contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson's disease patients that experience freezing of gait (FoG) during walking tasks.
  * [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)contains data of 6 different physical activities walking, walking upstairs, walking downstairs, sitting, standing and laying), performed by 30 subjects wearing a smartphone (Samsung Galaxy S II) on the waist.
  
## Apporach
  * for each dataset, a slicing window appoarch was applied to segment the dataset. Each segment includes a series of data (usually 25 sequential data points) and two continuous windows have 50% overlapping. 
  * After data preprocessing which includes reading files, data cleaning, data visualization, relabling and data segmentation, the data was saved into hdf5 files.
  * Deep learning models including CNN, DNN and RNN were applied. For each model in each dataset, hyperparameters were optimized to get the best performance.

## Dependencies
* Python 3.7
* tensorflow 1.13.1

## Usage
First download the dataset and put dataProcessing.py and models.py under the same directory. Then run dataprocessing to generate h5 file. Last switch model types in models.py script and run different deep learning models using the generated h5 data file.

## Note 
I am still working on tuning hyperparameters of models in certain datasets. There will be more updates. 

