# MediScan AI Model Training (YOLOv8)

This repository contains the training script for the core **Object Detection** feature of the **MediScan** application (AI Pharmacist for Seniors).

## 📌 Training Goals

This script performs fine-tuning based on the **YOLOv8 Nano (yolov8n)** model.

1. **Object Detection:** Identifies the location (**Bounding Box**) of pills within the camera frame.
2. **Lightweight Optimization (TFLite):** Converts the trained model into the **`.tflite` (TensorFlow Lite)** format for direct deployment on mobile devices (Flutter).
3. **Dataset:** Utilizes a custom pill dataset downloaded from Roboflow.

## 📂 Dataset

This project uses the **Pill Computer Vision Model** dataset from Roboflow Universe.
- **Source:** [Link to Roboflow Dataset](https://universe.roboflow.com/ang-jxyq0/pill-x13on )
- **License:** CC BY 4.0
- **Credit:** Special thanks to 'ang' for providing the dataset.

---

## 🛠️ Environment Setup

This guide assumes the use of a **Conda virtual environment** to enable GPU (CUDA) acceleration for faster training.

### 1. Prerequisites

* **Anaconda** or **Miniconda** installed.
* **NVIDIA GPU** with updated drivers installed.

### 2. Create Virtual Environment (Python 3.10 Recommended)

Open your terminal (PowerShell or CMD) and run the following commands in order:

```bash
# 1. Create a new environment (Name: mediscan_ai, Python version: 3.10)
conda create -n mediscan_ai python=3.10 -y

# 2. Activate the environment
conda activate mediscan_ai

```

### 3. Install PyTorch & CUDA (Crucial ⭐)

To ensure YOLOv8 utilizes the GPU, you must install the **CUDA-enabled version of PyTorch**. (Running a standard `pip install torch` may install the CPU-only version).

```bash
# Install PyTorch for NVIDIA CUDA 12.1 (Recommended for modern GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### 4. Install YOLOv8 & Required Libraries

Install the necessary libraries for training and TFLite conversion.

```bash
# Install Ultralytics (YOLOv8)
pip install ultralytics

# Install TensorFlow (Required for TFLite export - CPU version is fine)
pip install tensorflow

```

---

## 📂 Project Structure & Configuration

Before starting, verify your folder structure and the `data.yaml` configuration.

```text
MediScan/
├── dataset/               # Dataset downloaded from Roboflow
│   ├── train/
│   ├── valid/
│   └── data.yaml          # [IMPORTANT] Internal paths must be updated!
├── train.py               # Training script
└── yolov8n.pt             # (Automatically downloaded)

```

### ⚠️ Configuration Check (`data.yaml`)

Open `dataset/data.yaml` with a text editor and ensure that the `train` and `val` paths are set to **absolute paths** on your local machine.

```yaml
# Example (Windows Absolute Path, use forward slashes /)
train: D:/Project/MediScan/dataset/train/images
val: D:/Project/MediScan/dataset/valid/images

```

---

## 🚀 Usage

Once configured, run the following command to start training:

```bash
python train.py

```

### Execution Process

1. **GPU Check:** Verify that the message `✅ GPU Detected` appears at the start.
2. **Training:** The process will iterate through epochs (`Epoch 1/50`...).
3. **Export:** After training completes, the script will automatically export the model to `.tflite`.

---

## 📦 Output

Upon successful completion, the final model file will be generated at:

* **Path:** `runs/detect/medi_pill_model/weights/best_saved_model/`
* **Filename:** `best_float32.tflite`

### How to Apply to Flutter

1. Rename `best_float32.tflite` to **`yolov8n.tflite`**.
2. Copy the file to your Flutter project's `assets/models/` directory.
3. Create a `labels.txt` file, add the class name (e.g., `pill`), and place it in the same folder.

---

## ❓ Troubleshooting

**Q. I get a "Torch not compiled with CUDA enabled" error.**

* **A.** The CPU version of PyTorch is installed. Uninstall it and reinstall the CUDA version:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```



**Q. I get a "runtime error: path ... does not exist" error.**

* **A.** The paths in your `data.yaml` file are incorrect. Make sure you are using absolute paths (e.g., `D:/...`) and forward slashes (`/`).