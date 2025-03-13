# Sign Language Recognition Project

## 📌 Project Overview
This project is a **Sign Language Recognition System** that allows users to recognize hand signs using a trained deep learning model. It utilizes **TensorFlow** for model training and **OpenCV** for real-time video processing.

<!---
## 🎥 Demo
(Add a GIF or Screenshot of the application in action)
--->

## 🛠 Features
-  **Real-Time Hand Sign Recognition** - Uses a webcam to detect hand signs.
-  **Deep Learning Model** - Trained using a convolutional neural network (CNN).
-  **Alphabet Detection** - Recognizes 26 letters (A-Z) in sign language.
-  **Live Display** - Shows detected letters in real time.
-  **Interactive Controls** - Press 'Q' to exit the program.

---

## 📂 Project Structure
```
📂 SignLanguageProject
│── 📜 dataset.zip                   # Sign language dataset (preprocessed)
│── 📜 train_sign_language_model.py  # Model Training Script
│── 📜 real_time_sign_detection.py   # Real-Time Detection Script
│── 📜 sign_language_model.h5        # Trained Model File
│── 📜 README.md                     # Documentation File
```
## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies
```bash
pip install tensorflow opencv-python numpy pandas
```
### 2️⃣ Train the Model
```bash
python train_sign_language_model.py
```
### 3️⃣ Run Real-Time Detection
```bash
python real_time_sign_detection.py
```

---

## 🖐How It Works

### Training:
- Loads sign language images from the dataset  
- Processes and normalizes images  
- Trains a CNN to classify hand signs  
- Saves the trained model  

### Real-Time Detection:
- Captures frames from a webcam  
- Preprocesses frames to match model input  
- Predicts hand sign using the trained model  
- Displays the detected letter on the screen  

---

## 🏗Future Improvements
- Improve model accuracy with more diverse data  
- Implement multi-word recognition using hand gestures  
- Add support for dynamic sign gestures  



## 📜 License

This project is licensed under the MIT License.
