import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Load Model and Initialize MediaPipe
@st.cache_resource
def load_model():
    try:
        model = joblib.load('asl_model.joblib')
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

model = load_model()
hands, mp_drawing = init_mediapipe()

# Page config
st.set_page_config(page_title="EchoSign", page_icon="🤟", layout="wide")

st.title("🤟 EchoSign - Real-time ASL Translator")
st.markdown("""
### How to use:
1. Click **START** below to activate your webcam
2. Show an ASL letter sign (A-Z) to the camera
3. The predicted letter will appear in the **top-left corner** in large text
4. Make sure your hand is clearly visible in the frame
""")
st.divider()

# 2. Define the Video Transformer Class
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.hands, self.mp_drawing = init_mediapipe()
        self.prediction = None
        self.confidence = None
        self.hand_detected = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to NumPy array
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # Create overlay for UI elements
        overlay = image.copy()
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            self.hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with cleaner styling
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Normalization
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
                
                # Prediction
                try:
                    data_row = np.array(hand_data).reshape(1, -1)
                    prediction = self.model.predict(data_row)
                    predicted_letter = prediction[0]
                    
                    # Get confidence if available
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(data_row)
                        confidence = np.max(proba) * 100
                    else:
                        confidence = None
                    
                    self.prediction = predicted_letter
                    self.confidence = confidence
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.prediction = None
                    self.confidence = None
        else:
            self.hand_detected = False
            self.prediction = None
            self.confidence = None
        
        # Draw UI Elements
        # Top-left: Large prediction display (if hand detected)
        if self.hand_detected and self.prediction:
            box_w, box_h = 200, 200
            box_x, box_y = 20, 20
            
            # Semi-transparent dark background
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Green border
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 4)
            
            # Large letter display
            cv2.putText(image, self.prediction, (box_x + 35, box_y + 140), 
                       cv2.FONT_HERSHEY_BOLD, 5, (0, 255, 0), 10)
            
            # Confidence percentage (if available)
            if self.confidence:
                conf_text = f"{self.confidence:.0f}%"
                cv2.putText(image, conf_text, (box_x + 50, box_y + 185), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        # Top-right: Error message when no hand detected
        else:
            message_lines = [
                "No hand detected",
                "Please show your",
                "hand gesture clearly"
            ]
            
            box_w, box_h = 300, 130
            box_x, box_y = w - box_w - 20, 20
            
            # Semi-transparent red background
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            # Red border
            cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 3)
            
            # Warning emoji/icon
            cv2.putText(image, "!", (box_x + 15, box_y + 50), 
                       cv2.FONT_HERSHEY_BOLD, 2, (255, 255, 255), 4)
            
            # Message text
            for i, line in enumerate(message_lines):
                cv2.putText(image, line, (box_x + 60, box_y + 35 + i * 32), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bottom: Tips bar
        tips_h = 70
        cv2.rectangle(overlay, (0, h - tips_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
        
        # Tips text
        tips = [
            "TIPS: Keep hand centered | Good lighting helps | Try different angles",
        ]
        cv2.putText(image, tips[0], (20, h - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# 3. Run the Streamlit Web App
if model is None:
    st.error("❌ Model file 'asl_model.joblib' not found. Please run Phase 3 trainer first.")
else:
    st.success("✅ Model loaded successfully! Ready to recognize ASL letters A-Z")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_streamer(
            key="asl-translator",
            video_transformer_factory=ASLTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )
    
    st.divider()
    
    # Additional info
    with st.expander("ℹ️ Troubleshooting Tips"):
        st.markdown("""
        **If you're having issues:**
        - Make sure your browser has camera permissions enabled
        - Ensure your hand is well-lit and clearly visible
        - Keep your hand centered in the frame
        - Try holding the gesture steady for 1-2 seconds
        - Some letters may require practice to get the exact finger positioning
        
        **Best practices:**
        - Use a plain background for better hand detection
        - Keep the camera at arm's length
        - Make sure all fingers are visible (not cut off by frame edges)
        """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, MediaPipe, and scikit-learn")
