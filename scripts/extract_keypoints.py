import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Paths to the video folders and output CSV
GOOD_FORM_PATH = "C:/Users/Yash/Desktop/AI_Pushup_Tracking/data/good_form/"
BAD_FORM_PATH = "C:/Users/Yash/Desktop/AI_Pushup_Tracking/data/bad_form/"
OUTPUT_CSV = "C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/pushup_keypoints_reduced.csv"

# Define the keypoints we want to track (with ankle instead of foot)
KEYPOINTS = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,  # Replaced foot with ankle
    mp_pose.PoseLandmark.RIGHT_ANKLE  # Replaced foot with ankle
]

# Function to extract keypoints from a video
def extract_keypoints_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        result = pose.process(frame_rgb)

        # Extract specified pose landmarks
        if result.pose_landmarks:
            landmarks = []
            for idx in KEYPOINTS:
                lmk = result.pose_landmarks.landmark[idx]
                landmarks.append([lmk.x, lmk.y, lmk.z])
            landmarks = np.array(landmarks).flatten()
        else:
            landmarks = np.zeros(len(KEYPOINTS) * 3)  # 12 keypoints, 3 coordinates each (x, y, z)

        keypoints.append(np.append(landmarks, label))  # Add label to keypoints

    cap.release()
    return keypoints

# Process all videos in a folder and return their keypoints
def process_videos(input_path, label):
    all_keypoints = []
    for video in os.listdir(input_path):
        if video.endswith(".MOV") or video.endswith(".mp4"):
            print(f"Processing {video}...")
            video_path = os.path.join(input_path, video)
            keypoints = extract_keypoints_from_video(video_path, label)
            all_keypoints.extend(keypoints)
    return all_keypoints

# Main function to process all videos and save the keypoints to CSV
if __name__ == "__main__":
    # Extract keypoints for good and bad form videos
    good_keypoints = process_videos(GOOD_FORM_PATH, label=1)  # 1 for good form
    bad_keypoints = process_videos(BAD_FORM_PATH, label=0)   # 0 for bad form

    # Combine keypoints from both good and bad form videos
    all_data = np.array(good_keypoints + bad_keypoints)

    # Generate the header with the appropriate keypoints and label
    header = [f"{kp.name}_x, {kp.name}_y, {kp.name}_z" for kp in KEYPOINTS] + ["label"]
    
    # Save the keypoints and labels to a CSV file
    np.savetxt(OUTPUT_CSV, all_data, delimiter=",", header=",".join(header), comments="")
    print(f"Keypoints saved to {OUTPUT_CSV}")
