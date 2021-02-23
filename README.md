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
![Screenshots](mini_xception_plot.png)

