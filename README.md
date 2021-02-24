# Head-Tracking-and-Detection
## Head Pose Estimation:
Head Pose done through MTCNN Face Detector, tracking is done for yaw and pitch directions. 
Angles are calculated with the nose as the reference point, the distances from the eyes & mouth feature points with respect to the nose feature points helps calculate the yaw and pitch angles respectively. The head rotations also have labels to annotate which direction the head is rotated

## Eye Blink and Sleep Detection:
Dlib frontal face detector used to draw eye contours, the contours track the eye. 
The blink and sleep detections work based on EAR (Eye Aspect Ratio) threshold (set to 0.22), if average EAR (left and right eye) < 0.22 the eyes are considered closed. 
The blink duration is set to 1 frame and sleep is set to 5 frames. 

## Emotion Detection:
Haar Cascade frontal face detector is used for emotion recognition. 
Mini Xception model is trained on the on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
Below is the model accuracy plot:
![Accuracy Curves](https://github.com/vvenkatesh-ML/Head-Tracking-and-Detection-/blob/main/Screenshots/mini_xception_plot.png)

## Overall Summary:
![Summary Chart](https://github.com/vvenkatesh-ML/Head-Tracking-and-Detection-/blob/main/Screenshots/Vision%20System.png)

## How to Use:
The main file is 'head_pose_test.py' wihtin the 'src' folder, all required dependencies are within that folder. The 'head_pose_test.py' will output a video from the webcam which will also have all the above mentioned texts. 

For the emotion recognition, the trained model can be used directly if you wish to create a custome model use the 'data_preprocess.py' with the data attained from here: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv

The output video with all these features can be seen in the 'Output Video' folder



