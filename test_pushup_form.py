import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("/Users/kalp/Desktop/AI_Pushup_Tracking/AI-Pushup-Form-Tracker/model/pushup_model_augmented.h5")

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the keypoints you're using
keypoints = [
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
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# Initialize video capture
cap = cv2.VideoCapture("/Users/kalp/Desktop/AI_Pushup_Tracking/AI-Pushup-Form-Tracker/test_video/test_video3.mp4")

# Initialize variables to track pushups
rep_count = 0
in_pushup = False  # To track whether we're in the middle of a pushup
is_good_form = False
top_position_reached = False  # To ensure we only count at the top
bottom_position_reached = False  # To track when we reach the bottom
good_top_position = False  # To track good form at the top
good_bottom_position = False  # To track good form at the bottom

# Variables to track the dynamic range of shoulder positions
shoulder_min = float('inf')
shoulder_max = float('-inf')

# Flags to control counting
first_rep_started = False  # To track if the first rep has started
first_rep_thresholds_set = False  # To track if the first rep thresholds are set
frame_buffer = 10  # Number of frames to wait before starting to count pushups
frame_counter = 0  # Counter to track frame buffer progress

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Check if keypoints are detected
    if results.pose_landmarks:
        # Extract the keypoints
        keypoints_list = []
        for landmark in keypoints:
            keypoint = results.pose_landmarks.landmark[landmark]
            keypoints_list.extend([keypoint.x, keypoint.y, keypoint.z])

        # Convert the keypoints to a numpy array and reshape it
        keypoints_array = np.array(keypoints_list).reshape(1, -1)

        # Predict the form (Good or Bad)
        prediction_prob = model.predict(keypoints_array)
        prediction_class = "Good" if prediction_prob > 0.5 else "Bad"
        
        # Update good form status
        is_good_form = prediction_class == "Good"

        # Get shoulder positions to detect pushup motion
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2  # Average shoulder position

        # Calculate static thresholds for the first rep
        if not first_rep_thresholds_set:
            shoulder_min = shoulder_y
            shoulder_max = shoulder_y
            first_rep_thresholds_set = True

        # Update the dynamic shoulder min and max values after the first rep
        shoulder_min = min(shoulder_min, shoulder_y)
        shoulder_max = max(shoulder_max, shoulder_y)

        # Calculate the dynamic range for top and bottom positions
        shoulder_range = shoulder_max - shoulder_min
        top_threshold = shoulder_min + (shoulder_range * 0.1)  # Top range based on min position
        bottom_threshold = shoulder_max - (shoulder_range * 0.1)  # Bottom range based on max position

        # Track the first rep logic
        if frame_counter < frame_buffer:
            frame_counter += 1
            continue  # Skip counting for the first few frames

        # Detect pushup positions (top -> bottom -> top)
        if shoulder_y < top_threshold:  # At the top position
            if not top_position_reached:
                top_position_reached = True
                good_top_position = is_good_form  # Store if form is good at top

        elif shoulder_y > bottom_threshold:  # At the bottom position
            if top_position_reached and not bottom_position_reached:
                bottom_position_reached = True
                good_bottom_position = is_good_form  # Store if form is good at bottom

        # Count the pushup only when top -> bottom -> top with good form
        if top_position_reached and bottom_position_reached and shoulder_y < top_threshold:  # Back at top
            if good_top_position and good_bottom_position:  # Only count if both top and bottom had good form
                rep_count += 1
            # Reset only the bottom-related flags for next rep
            bottom_position_reached = False
            good_bottom_position = False
            top_position_reached = False  # Reset top after counting the rep

        # Display the count of good pushups
        cv2.putText(frame, f"Pushups: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the form prediction
        prediction_text = f"{prediction_class} Form: {prediction_prob[0][0]*100:.2f}%"
        color = (0, 255, 0) if prediction_class == "Good" else (0, 0, 255)
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame with the prediction
    cv2.imshow("Pushup Form", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()