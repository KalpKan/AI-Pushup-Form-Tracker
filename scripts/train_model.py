import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset (replace with your path)
data = pd.read_csv("C:/Users/Yash/Desktop/AI_Pushup_Tracking/outputs/pushup_keypoints_augmented.csv")

# Split the data into features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last one (the keypoints)
y = data.iloc[:, -1].values   # The last column is the label (0 for bad, 1 for good form)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance in neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]))  # 36 features (keypoints)
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[early_stopping])

# Save the trained model
model.save("C:/Users/Yash/Desktop/AI_Pushup_Tracking/model/pushup_form_model.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
