import cv2
import mediapipe as mp
import numpy as np
import joblib

# 1. Load the trained model
try:
    model = joblib.load('asl_model.joblib')
except FileNotFoundError:
    print("ERROR: Model file 'asl_model.joblib' not found.")
    print("Please run the trainer script (Phase 3) first.")
    exit()

# 2. Initialize MediaPipe and Webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting Real-time ASL Translator... Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip and convert color
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(rgb_image)

    # This will hold our 42 data points for the model
    normalized_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks for visual feedback
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # --- Start Normalization (Same as Phase 2) ---
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append([landmark.x, landmark.y])
            
            base_x, base_y = landmark_list[0]
            
            hand_data = []
            for landmark_coords in landmark_list:
                relative_x = landmark_coords[0] - base_x
                relative_y = landmark_coords[1] - base_y
                hand_data.append(relative_x)
                hand_data.append(relative_y)
                
            normalized_landmarks = hand_data
            # --- End Normalization ---
            
            # --- Start Prediction ---
            if normalized_landmarks:
                # Convert to NumPy array and reshape
                data_row = np.array(normalized_landmarks).reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(data_row)
                predicted_letter = prediction[0]
                
                # --- Display the Prediction ---
                # Get the coordinates of the wrist (landmark 0) to draw text near the hand
                wrist_x = int(hand_landmarks.landmark[0].x * image.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * image.shape[0])

                cv2.putText(
                    image,
                    f"Prediction: {predicted_letter}",
                    (wrist_x - 50, wrist_y - 50), # Position text above the wrist
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,                 # Font scale
                    (0, 255, 0),       # Color (Green)
                    2                  # Thickness
                )
            # --- End Prediction ---
    
    # Show the image
    cv2.imshow('Real-time ASL Translator - Phase 4', image)

    # Quit logic
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Translator stopped.")
