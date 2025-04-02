import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("C:/Users/Yash/Desktop/AI_Pushup_Tracking/model/pushup_form_model.h5")

# Load the new data to test or predict
data = pd.read_csv("C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/new_keypoints.csv")

# Preprocess the data (scaling it)
X_new = data.iloc[:, :-1].values  # All columns except the last one (the keypoints)
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)

# Print the predictions (0 = bad form, 1 = good form)
for idx, pred in enumerate(predictions):
    print(f"Frame {idx + 1}: {'Good Form' if pred >= 0.5 else 'Bad Form'}")
