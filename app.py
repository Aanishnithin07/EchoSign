import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

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
    # Optimize for performance: lower confidence, disable model complexity
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # Use lightweight model (0 = fastest)
    )
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
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Process with MediaPipe directly on original resolution
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        if results.multi_hand_landmarks:
            self.hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Simple landmark drawing
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

        if results.multi_hand_landmarks:
            self.hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Simple landmark drawing
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
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
                    
                    # Draw prediction box with solid background
                    # Filled rectangle background
                    cv2.rectangle(img, (10, 10), (160, 160), (0, 0, 0), -1)  # Black filled
                    # Green border
                    cv2.rectangle(img, (10, 10), (160, 160), (0, 255, 0), 4)
                    # Large letter in white for visibility
                    cv2.putText(img, predicted_letter, (35, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)
                    # Confidence below
                    if confidence:
                        cv2.putText(img, f"{confidence:.0f}%", (45, 145), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.prediction = None
                    self.confidence = None
        else:
            self.hand_detected = False
            self.prediction = None
            self.confidence = None
            
            # "No hand" message with background
            cv2.rectangle(img, (w - 310, 15), (w - 10, 65), (0, 0, 0), -1)  # Black background
            cv2.rectangle(img, (w - 310, 15), (w - 10, 65), (0, 0, 255), 3)  # Red border
            cv2.putText(img, "Show hand gesture", (w - 295, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Run the Streamlit Web App
if model is None:
    st.error("❌ Model file 'asl_model.joblib' not found. Please run Phase 3 trainer first.")
else:
    st.success("✅ Model loaded successfully! Ready to recognize ASL letters A-Z")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_ctx = webrtc_streamer(
            key="asl-translator",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=ASLTransformer,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "frameRate": 30
                },
                "audio": False
            },
            async_transform=False,  # Changed to sync for better stability
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    
    st.divider()
    
    # Performance tips
    st.info("💡 **Performance Tip**: If the video is laggy, try refreshing the page or restarting the webcam stream by clicking STOP then START again.")
    
    # Additional info
    with st.expander("ℹ️ Troubleshooting Tips"):
        st.markdown("""
        **If you're having issues:**
        - **Frozen/Laggy video**: Click STOP, wait 2 seconds, then click START again
        - Make sure your browser has camera permissions enabled (check browser settings)
        - Close other applications using the webcam
        - Try a different browser (Chrome or Edge work best)
        - Ensure your hand is well-lit and clearly visible
        - Keep your hand centered in the frame
        - Try holding the gesture steady for 1-2 seconds
        - Some letters may require practice to get the exact finger positioning
        
        **Best practices:**
        - Use a plain background for better hand detection
        - Keep the camera at arm's length
        - Make sure all fingers are visible (not cut off by frame edges)
        - Good lighting is essential for accurate detection
        """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, MediaPipe, and scikit-learn")
