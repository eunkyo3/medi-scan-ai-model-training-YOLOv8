from ultralytics import YOLO
import torch
import os

def main():
    print("-" * 50)
    print("Initializing Training Script...")
    
    # 1. Check if GPU is available
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {device_name}")
        print("Training will run on GPU (Fast).")
    else:
        print("⚠️ GPU Not Detected.")
        print("Training will run on CPU (Slower but stable with Ryzen 5700X3D).")
    print("-" * 50)

    # 2. Load the Model
    # We use 'yolov8n.pt' (Nano), which is the smallest and fastest model for mobile.
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 

    # 3. Train the Model
    # IMPORTANT: You MUST change the 'data' path below to your actual 'data.yaml' path.
    # Use absolute path (e.g., 'D:/Project/MediScan/dataset/data.yaml') to avoid errors.
    print("Starting training...")
    try:
        results = model.train(
            data=r'D:\pill_yolov8\Pill.v1i.yolov8/data.yaml',  # <--- UPDATE THIS PATH!
            epochs=50,           # 50 epochs is usually enough for simple objects like pills
            imgsz=640,           # Standard YOLOv8 image size
            batch=16,            # Reduce to 8 or 4 if you run out of memory
            name='medi_pill_model', # Name of the folder where results will be saved
            exist_ok=True,       # Overwrite existing folder if it exists
            device=0 if torch.cuda.is_available() else 'cpu' # Force device selection
        )
        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # 4. Export to TFLite (For Flutter)
    print("-" * 50)
    print("Exporting model to TFLite format...")
    try:
        # Exporting to TFLite format for mobile deployment
        # The file 'best_float32.tflite' will be generated
        model.export(format='tflite')
        
        print("✅ Export completed successfully!")
        print(f"Check your results in: runs/detect/medi_pill_model/weights/")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")

if __name__ == '__main__':
    # This block is required for Windows execution to prevent multiprocessing errors
    main()