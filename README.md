cat > README.md << 'EOF'
# Cattle & Buffalo Breed Recognition (SIH 2025)

## 📌 Problem Statement
Image-based breed recognition for cattle and buffaloes of India.  
Proposed as part of **Smart India Hackathon 2025** (Problem Statement ID – 25004).

## 🚀 Proposed Solution
- Mobile-first AI tool integrated into the Bharat Pashudhan App (BPA).
- FLW (Field-Level Worker) snaps a picture → AI predicts breed in real-time.
- Suggests/confirms breed during registration, eliminating manual errors.
- Works both **offline (TFLite)** and **online (FastAPI cloud API)**.

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras (MobileNetV2 + Transfer Learning)**
- **TensorFlow Lite** (for mobile deployment)
- **FastAPI** (for cloud inference API)
- **scikit-learn** (metrics & evaluation)
- **OpenCV** (image handling)

## 📂 Project Structure
\`\`\`
cattle-breed-recognition/
│
├── README.md
├── requirements.txt
├── train_baseline.py      # baseline MobileNetV2 training
├── eval.py                # evaluate model (accuracy, precision, recall, F1, confusion matrix)
├── export_tflite.py       # convert trained model to TFLite
├── server.py              # FastAPI inference server
└── dataset/               # dataset folder (not uploaded to repo)
    ├── train/
    │   ├── breed_A/
    │   ├── breed_B/
    │   └── ...
    ├── val/
    │   ├── breed_A/
    │   └── ...
    └── test/
        ├── breed_A/
        └── ...
\`\`\`

## ⚡ Setup Instructions
Clone the repository:
\`\`\`bash
git clone https://github.com/<your-username>/cattle-breed-recognition.git
cd cattle-breed-recognition
\`\`\`

Create virtual environment & install dependencies:
\`\`\`bash
python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows

pip install -r requirements.txt
\`\`\`

## 🧪 Workflow
1. **Data Collection** – Collect Indian cattle & buffalo breed images.
2. **Training** – Run \`train_baseline.py\` (MobileNetV2 transfer learning).
3. **Evaluation** – Run \`eval.py\` to compute accuracy, precision, recall, F1, confusion matrix.
4. **Export** – Convert trained model to TFLite using \`export_tflite.py\`.
5. **Deployment** – 
   - On-device: TFLite model inside BPA app.  
   - Cloud: FastAPI server (\`server.py\`).

## 🎯 Impact
- ✅ Eliminates manual errors in breed recognition.  
- ✅ Empowers field workers with real-time AI support.  
- ✅ Provides reliable data for government livestock planning.  
- ✅ Boosts farmers’ income with better breeding & healthcare programs.

## 📖 References
- Research papers on animal breed classification using CNNs.  
- TensorFlow Hub models (MobileNetV2).  
- Kaggle datasets.  
- Studies on Indian cattle & buffalo breed diversity.

---

👨‍💻 Developed by **Team MaxPool Mavericks**  
for **Smart India Hackathon 2025**
EOF

