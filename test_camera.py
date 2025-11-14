import cv2

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Camera opened successfully!")
print("Press 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"ERROR: Failed to grab frame {frame_count}")
        break
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")
    
    frame = cv2.flip(frame, 1)
    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Test complete. Total frames: {frame_count}")
