# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the MobileNetV2 pre-trained model
model = MobileNetV2(weights='imagenet')

# Open webcam for capturing real-time video
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to preprocess the frame for MobileNetV2
def preprocess_frame(frame):
    # Resize frame to 224x224 as required by MobileNetV2
    img_resized = cv2.resize(frame, (224, 224))
    
    # Convert image from BGR (OpenCV format) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a numpy array and expand dimensions to match the input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Preprocess the image for the MobileNetV2 model
    return preprocess_input(img_array)

# Loop for real-time classification
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    # If frame is not captured correctly, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame for model input
    processed_frame = preprocess_frame(frame)

    # Perform prediction
    preds = model.predict(processed_frame)

    # Decode the predictions to get the class label and confidence score
    decoded_preds = decode_predictions(preds, top=1)[0]

    # Extract the most probable class name and confidence
    label = decoded_preds[0][1]
    confidence = decoded_preds[0][2] * 100

    # Display the prediction on the video feed
    text = f'{label}: {confidence:.2f}%'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with predictions
    cv2.imshow('Real-Time Object Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
