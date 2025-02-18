import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set parameters
frame_height = 64
frame_width = 64
frame_count = 30  # Number of frames per sequence
video_dir_normal = "C:/Users/D.V. kusumanjali/OneDrive/Desktop/Dataset/Dataset/Normal"
video_dir_shoplifting = "C:/Users/D.V. kusumanjali/OneDrive/Desktop/Dataset/Dataset/Shoplifting"

# Function to preprocess frames from videos
def preprocess_video(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        frame_normalized = frame_resized / 255.0  # Normalize pixel values to [0, 1]
        frames.append(frame_normalized)
        
        if len(frames) == frame_count:
            break  # Only use the first 'frame_count' frames for the sequence

    video.release()
    
    # If there are fewer than frame_count frames, pad with zeros
    while len(frames) < frame_count:
        frames.append(np.zeros((frame_height, frame_width, 3)))

    return np.array(frames)

# Function to load dataset from given directories
def load_dataset():
    X = []  # Feature list (frames)
    y = []  # Label list (0 for Normal, 1 for Shoplifting)
    
    # Load normal videos
    for i in range(1, 91):  # Normal videos from 1 to 90
        video_path = os.path.join(video_dir_normal, f"Normal ({i}).mp4")
        frames = preprocess_video(video_path)
        X.append(frames)
        y.append(0)  # Label 0 for normal
    
    # Load shoplifting videos
    for i in range(1, 94):  # Shoplifting videos from 1 to 93
        video_path = os.path.join(video_dir_shoplifting, f"Shoplifting ({i}).mp4")
        frames = preprocess_video(video_path)
        X.append(frames)
        y.append(1)  # Label 1 for shoplifting
    
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, num_classes=2)  # One-hot encode the labels
    
    return X, y

# Load the dataset
X, y = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(frame_count, frame_height, frame_width, 3)))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D((2, 2, 2)))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D((2, 2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 classes: normal and shoplifting

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=85,batch_size=8, validation_data=(X_test, y_test))

# Save the trained model
model.save('model.h5')
print("Model saved as model.h5")
