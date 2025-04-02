import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path to the folder containing "Good Form" videos
good_form_folder = 'C:/Users/Yash/Desktop/AI_Pushup_Tracking/data/good_form'

# List to hold all keypoints
keypoints_data = []

# Extract keypoints from each video
for video_file in os.listdir(good_form_folder):
    video_path = os.path.join(good_form_folder, video_file)
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        keypoints = []
        if results.pose_landmarks:
            # Extract only the relevant keypoints (12 keypoints you specified)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z)
            
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
            keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z)
            
            keypoints_data.append(keypoints)

    cap.release()

# Create a DataFrame for the keypoints
df = pd.DataFrame(keypoints_data, columns=[f'keypoint_{i}' for i in range(1, len(keypoints_data[0])+1)])
df['label'] = 'Good'  # Since we are training only on good form videos

# Save to a CSV file
df.to_csv('good_form_keypoints.csv', index=False)
print("Keypoints extracted and saved to 'good_form_keypoints.csv'.")
