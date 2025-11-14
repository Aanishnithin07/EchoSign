---
title: EchoSign - ASL Recognition
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.51.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# 🤟 EchoSign - Real-time ASL Recognition

EchoSign is a real-time American Sign Language (ASL) recognition system that uses computer vision and machine learning to translate hand gestures into letters (A-Z).

## ✨ Features

- 🎥 Real-time webcam-based hand tracking
- 🤖 97.32% accurate ML model using Random Forest
- 🌐 Web-based interface with Streamlit
- 📊 Live confidence scores
- 🎯 Optimized for performance with MediaPipe Hands

## 🚀 Live Demo

**[Try EchoSign Live on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/EchoSign)** _(Update with your HF username after deployment)_

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe Hands
- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Web Framework**: Streamlit, streamlit-webrtc
- **Data Processing**: NumPy, Pandas

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Aanishnithin07/EchoSign.git
cd EchoSign
```

2. Create a virtual environment (Python 3.12):
```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate  # On Windows: .venv-py312\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Git LFS (if cloning the model):
```bash
git lfs install
git lfs pull
```

## 🎮 Usage

Run the Streamlit app locally:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How It Works

1. **Hand Tracking**: MediaPipe detects 21 hand landmarks in real-time
2. **Feature Extraction**: Landmark coordinates are normalized and flattened
3. **Prediction**: Random Forest model classifies the hand gesture
4. **Display**: Predicted letter and confidence score shown on screen

## 📊 Model Performance

- **Accuracy**: 97.32%
- **Training Samples**: 1,864 gestures
- **Features**: 42 normalized landmark coordinates
- **Classes**: 26 ASL letters (A-Z)

## 🏗️ Project Structure

```
EchoSign/
├── app.py                    # Streamlit web application
├── phase1_hand_tracker.py    # Hand tracking module
├── phase2_data_collector.py  # Data collection tool
├── phase3_train_model.py     # Model training script
├── phase4_realtime_test.py   # Real-time testing
├── asl_model.joblib          # Trained ML model (Git LFS)
├── asl_dataset.csv           # Training dataset
└── requirements.txt          # Python dependencies
```

## 🔧 Development Phases

1. **Phase 1**: Hand tracking with MediaPipe
2. **Phase 2**: Data collection (30 samples per letter)
3. **Phase 3**: Model training with Random Forest
4. **Phase 4**: Real-time testing
5. **Phase 5**: Web deployment with Streamlit
6. **Phase 6**: Cloud deployment

## 🌐 Deployment

Deployed on Streamlit Cloud for free public access.

## 📝 License

MIT License - feel free to use this project for learning and development!

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 👨‍💻 Author

**Aanish Nithin**
- GitHub: [@Aanishnithin07](https://github.com/Aanishnithin07)

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools

---

⭐ Star this repo if you found it helpful!
