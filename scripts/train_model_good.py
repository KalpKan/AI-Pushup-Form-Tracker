import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the augmented dataset
input_file_path = "C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/augmented_good_form_keypoints.csv"
data = pd.read_csv(input_file_path)

# Strip extra spaces from column names
data.columns = data.columns.str.strip()

# Split the data into features (X) and labels (y)
X = data.drop('label', axis=1).values  # Features (keypoints)
y = data['label'].values  # Labels (good/bad form)

# Normalize the feature data (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model for future use
model.save("good_form_model.h5")
print("Model saved as 'good_form_model.h5'")
