# real_time_sign_detection.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load the pre-trained model
model = load_model('sign_language_model.h5')

# Define a dictionary to map the model's output to sign language letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model
    frame_resized = cv2.resize(frame, (64, 64))  # Assuming model input size is 64x64
    frame_normalized = frame_resized / 255.0  # Normalize the frame
    return np.expand_dims(frame_normalized, axis=0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(processed_frame)
    predicted_label = np.argmax(predictions)

    # Get the corresponding sign language letter
    detected_sign = labels_dict[predicted_label]

    # Display the resulting frame with the detected sign
    cv2.putText(frame, f'Sign: {detected_sign}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Sign Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()
