# EchoSign 🤟: Real-time ASL Translator

> An AI-powered communication bridge that translates American Sign Language (ASL) into text, in real-time, directly in your browser.

[![Streamlit App](https://img.shields.io/badge/LIVE_DEMO-blue?style=for-the-badge&logo=Streamlit)](https://YOUR_STREAMLIT_APP_URL.streamlit.app)
[![Repo Status](https://img.shields.io/badge/Status-Complete-green?style=for-the-badge)](https://github.com/Aanishnithin07/EchoSign)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen?style=for-the-badge)](https://opensource.org/licenses/MIT)

---



---

## Core Features

* **Real-time Translation:** Instant classification of hand gestures from a live webcam feed.
* **High-Accuracy Model:** 97.3% test accuracy using a **`Random Forest Classifier`**.
* **Robust & Invariant:** The feature engineering makes the model resilient to changes in hand size, position, and rotation.
* **Web-Based Interface:** Deployed as a public-facing Streamlit app. No install needed.
* **Lightweight & Fast:** Built on the high-performance **`MediaPipe`** framework for tracking.

---

## The Tech Stack

This project isn't just a simple script; it's an integrated system of modern AI and web technologies.

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Machine Learning** | **Scikit-learn** | For training the `RandomForestClassifier` model. |
| **Computer Vision** | **MediaPipe** (by Google) | For high-fidelity, real-time hand and 21-point landmark detection. |
| | **OpenCV** | For image processing, text rendering, and video stream handling. |
| **Web Framework** | **Streamlit** | To build and deploy the interactive web application. |
| | **Streamlit-WebRTC** | To handle the real-time webcam feed directly in the browser. |
| **Data Handling** | **NumPy** & **Pandas** | For landmark data manipulation, normalization, and CSV handling. |
| **Model Storage** | **Git LFS** | To manage the large `asl_model.joblib` file in the repository. |

---

## The Engineering: How It Works

The "magic" is a four-step pipeline, with the critical step being **Feature Normalization**.

1.  **Capture & Detect:** `Streamlit-WebRTC` grabs the webcam frame. `MediaPipe` scans the frame and detects the 21 3D landmarks of the hand.

2.  **The "Secret Sauce" - Normalization:** The raw coordinates are useless. A hand close to the camera has different coordinates than one far away. The solution is **normalization**.
    * The wrist (landmark 0) is set as the origin `(0, 0)`.
    * All other 20 landmarks are recalculated as vectors *relative to the wrist*.
    * This 42-point vector (`21 * 2 coordinates`) is now **invariant** to position.

3.  **Predict:** This normalized vector is fed into the pre-trained `RandomForestClassifier` (`asl_model.joblib`), which instantly returns a prediction (e.g., "A", "B", "L").

4.  **Display:** The predicted letter is rendered back onto the video frame using `OpenCV`, and the frame is displayed to the user.

---

## 🚀 Run It Locally

Want to run the full pipeline on your own machine?

**Prerequisites:** Python 3.10+, Git, and **Git LFS**.

1.  **Clone the repo (LFS is required):**
    ```bash
    git lfs install
    git clone [https://github.com/Aanishnithin07/EchoSign.git](https://github.com/Aanishnithin07/EchoSign.git)
    cd EchoSign
    git lfs pull
    ```
    *(Why LFS? The `asl_model.joblib` file is too large for standard Git.)*

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    Your browser will open to `http://localhost:8501`.

---

## 👨‍💻 Author

* **Aanish Nithin A**
* [GitHub: @Aanishnithin07](https://github.com/Aanishnithin07)
⭐ **Star this repo if you found it helpful.**
