# Consentric and Eccentric Squat Prediction 
This project is the code to take the prediction from YOLO feed it into media pipe and use that as features for machine learning models.

# Data
All the video samples we collected is under the AllVids folder. The videos were run through PoseModYolo to create the full data set that was split up into the test and train set. 

# How To Run 
1. Clone this repo
2. Most of what you need is already in the repo but the yolo weights are too big to be stored in github so go to [The Yolo github](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights) and just download the yolo3.weights and put it in the file called YOLO. 


### Dependencies List

Install the required packages using:
```bash
pip install numpy opencv-python torch scikit-learn xgboost joblib ultralytics mediapipe

```

3. After you set up your virtual environment ( I used python3.11 to run this) and pip install the libs above, run the file PredictPose.py. The video it predicts is in main. Feel free to change the video path to some of the files we stored in AllVids.  
