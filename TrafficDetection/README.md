# 🚦 Traffic Detection with Ensemble YOLOv11 & Docker 🐳

This project performs real-time **traffic object detection** using an **ensemble of two YOLOv11 models**. The ensemble approach enhances detection accuracy by combining predictions. The project is containerized with **Docker** for ease of deployment.

---

## 🧠 Features

- 🧩 **Ensemble Learning** of two YOLOv11 models  
- 🚗 Real-time detection of vehicles, pedestrians, traffic signs, and more  
- 🐳 **Dockerized** for portability and reproducibility  


---

## 📁 Project Structure

- `Data/` : datasets
- `models/` : trained models
- `notebooks/` : Jupyter notebooks for training and evaluation
---

## ⚙️ Ensemble Logic

The ensemble model merges the outputs from two YOLOv11 detectors by applying:

- **Non-Maximum Suppression (NMS) across models**
- Optional **voting or averaging** of confidence scores  
- Improved precision and recall over a single model

---

## 🐳 Docker Setup

### 1️⃣ Build the Docker Image

```bash
docker build -t traffic-detector .

docker run -v "PATH":/data traffic-detection  --input INPUT  --output OUTPUT

