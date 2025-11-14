import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- Setup from Phase 1 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- New for Phase 2 ---
DATASET_FILE = 'asl_dataset.csv'
NUM_LANDMARKS = 21

# Create the header for our CSV file
# We need 'label' (for A, B, C...)
# Then 21 * 2 (x, y) coordinates = 42 columns
header = ['label']
for i in range(NUM_LANDMARKS):
    header += [f'x{i}', f'y{i}']

# Write header only if file doesn't exist or is empty
if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
    with open(DATASET_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

print("Starting Data Collector... Press A-Z to save data. Press 'q' to quit.")

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_image)
    
    # We'll store the normalized data here
    normalized_landmarks = []

    # Draw the hand annotations and extract landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Draw landmarks for visual feedback
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # 2. Start Normalization
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append([landmark.x, landmark.y])
            
            # 3. Get base (wrist) coordinate
            base_x, base_y = landmark_list[0]
            
            # 4. Create a list for normalized coordinates
            hand_data = []
            
            for landmark_coords in landmark_list:
                # 5. Calculate relative coordinates
                relative_x = landmark_coords[0] - base_x
                relative_y = landmark_coords[1] - base_y
                hand_data.append(relative_x)
                hand_data.append(relative_y)
                
            # Now, 'hand_data' contains 42 numbers (21 x, y pairs)
            # all relative to the wrist.
            normalized_landmarks = hand_data
    
    # Display the image
    cv2.imshow('MediaPipe Data Collector - Phase 2', image)

    # Key logic
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):
        break
    
    # Check if the key is a letter (a-z)
    if (key >= ord('a') and key <= ord('z')):
        # And check if we actually detected a hand
        if normalized_landmarks:
            label = chr(key).upper() # Store as uppercase 'A'
            
            # Prepend the label to our data row
            data_row = [label] + normalized_landmarks
            
            # Append to the CSV file
            with open(DATASET_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
                
            print(f"SUCCESS: Saved data for letter '{label}'")
        else:
            print("WARNING: No hand detected. Data not saved.")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Data collection stopped.")
