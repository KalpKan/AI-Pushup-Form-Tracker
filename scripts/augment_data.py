import numpy as np
import pandas as pd

def augment_keypoints(keypoints, max_deviation=0.05):
    """Augment keypoints by adding small deviations."""
    return keypoints + np.random.uniform(-max_deviation, max_deviation, keypoints.shape)

def augment_data(input_csv, output_csv):
    # Load the dataset
    data = pd.read_csv(input_csv)
    
    # Augment the keypoints (excluding the label column)
    augmented_keypoints = []
    for index, row in data.iterrows():
        keypoints = row[:-1].values  # Exclude the label column
        augmented_keypoints.append(augment_keypoints(keypoints))
    
    augmented_data = pd.DataFrame(augmented_keypoints)
    augmented_data['label'] = data['label']
    
    # Save the augmented data to a new CSV
    augmented_data.to_csv(output_csv, index=False)

# Example of augmenting data
augment_data('C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/pushup_keypoints_reduced.csv', 
             'C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/pushup_keypoints_augmented.csv')
