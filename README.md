# Real-time-face-mask-detection

This repository contains python code to detect faces with or without masks. 

Concept of DeepNeuralNetworks is used to prepare the model to detect the faces with or without masks. And, to prepare the model I've used `ResNet50_v2` architecture to prepare the CNN.

`OpenCV` is used to detect faces in real-time using the Video Stream from the WebCam.

### Requirements
- Tensorflow(>2.0)
- [ResNet50_v2](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/#:~:text=ResNet%2C%20which%20was%20proposed%20in%202015%20by%20researchers,we%20use%20a%20technique%20called%20skip%20connections%20.)
- OpenCV

### [Google Colab notebook](https://colab.research.google.com/drive/1CYFml4fKY137DYGM_Kr0CqkbLX4pB8ea?usp=sharing)

[Dataset](https://data-flair.s3.ap-south-1.amazonaws.com/Data-Science-Data/face-mask-dataset.zip)
### to use the above dataset, follow the steps below
- extract the "face-mask-dataset.zip" into the main project folder
- go to the directory called "Dataset"(created after following above step)
- unzip "train.zip" and "test.zip"( You can delete these zip files after extracting, to save space)

### How to start Detection -
run "detection.py"

### To use this repository with Bash
#### 1. Download the repository
```
git clone https://github.com/Kamlesh364/Real-time-face-mask-detection.git
```

#### 2. Download the dataset
```
!wget https://data-flair.s3.ap-south-1.amazonaws.com/Data-Science-Data/face-mask-dataset.zip
```

#### 3. prepare the dataset
- unzipping dataset
```
unzip face-mask-dataset.zip
```
- unzipping testing and training datasets
```
cd Dataset 
unzip train.zip 
unzip test.zip 
rm -rf train.zip test.zip 
cd .. 
```
- train the model to test it in real-time
```
python3 train.py
```
- test the model in real-time
```
python3 test.py
```

### Working demo
![Alt text](face-mask-detector-project.gif "Signal processing")
