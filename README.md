# Linguisync-3D: Neural Lip & Facial Dubbing

**Cross-Modal 3D-Aware Neural Dubbing System using Joint Audio-Visual Embedding**

---

## 🎯 Project Overview

**Linguisync-3D** is a deep learning-based neural facial dubbing system that can dub videos into different languages while maintaining natural lip movements and facial expressions. 

Unlike traditional 2D pixel-warping methods, this project focuses on **3D facial geometry awareness** and **cross-modal temporal consistency** using a custom Joint Audio-Visual Embedding Loss.

---

## ✨ Key Features

- **3D Facial Awareness**: Extracts 478 3D landmarks using MediaPipe
- **Rich Audio Understanding**: Uses Wav2Vec 2.0 for phoneme-level features
- **Novel Joint Sync Loss**: Aligns audio and 3D visual embeddings frame-by-frame
- **Base Generator**: Wav2Lip for high-quality lip synthesis
- **Temporal Smoothing**: Reduces jitter between frames
- **Fully Offline**: No external APIs required
- **Multilingual Support**: Tested with English & Hindi

---

## 🛠️ Tech Stack

| Component                | Technology                     |
|-------------------------|--------------------------------|
| Audio Feature Extraction| Wav2Vec 2.0 (Hugging Face)     |
| 3D Landmarks            | MediaPipe Face Mesh            |
| Lip Sync Generator      | Wav2Lip                        |
| Joint Embedding         | Custom PyTorch NN              |
| Framework               | PyTorch + OpenCV               |
| Environment             | Google Colab (Free Tier)       |

---

## 📁 Project Structure
Linguisync3D/
├── data/                      # GRID Dataset videos
├── results/                   # Output videos & models
├── src/                       # (Optional) Source code
├── notebooks/                 # Colab notebooks
├── improved_joint_embedder.pth # Trained model
└── README.md


---

## 🚀 How It Works (Pipeline)

1. Input video + Target audio (any language)
2. Extract 3D facial landmarks (MediaPipe)
3. Extract audio features (Wav2Vec 2.0)
4. Compute **Joint Audio-Visual Sync Loss** (Main Contribution)
5. Generate lip movements using Wav2Lip
6. Apply temporal smoothing
7. Output: Natural dubbed video

**Core Novelty**:
```math
<img width="434" height="86" alt="image" src="https://github.com/user-attachments/assets/d60325df-78ea-4783-b8c8-6d4f19e8bcac" />


📊 Results
Successfully dubbed multiple English & Hindi audios
Joint Sync Loss achieved: 0.0152 (after training)
Working with long paragraphs and energetic speech
Temporal smoothing applied for better visual quality

🔮 Future Work
Full integration with FLAME 3DMM
End-to-end fine-tuning of generator
Real-time dubbing using MuseTalk
Evaluation on LRS2 / MEAD datasets
User study for MOS score

👨‍💻 Team / Author
Kavya Thakar
Department of Information and Communication Technology
Marwadi University, Rajkot

Made with ❤️ using PyTorch, MediaPipe & Wav2Lip

How to Use
Clone / Open in Colab
Mount Google Drive
Run the main pipeline notebook
Put your video and audio in proper folders
Run dubbing script
