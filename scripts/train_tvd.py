import os
from ultralytics import YOLO
import torch

def main():
    # Print hardware info
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {'GPU' if device == '0' else 'CPU'}")

    # Load a pre-trained base model for transfer learning
    # yolo11s is a good balance between inference speed and accuracy
    print("Loading YOLO11s base model...")
    model = YOLO('yolo11s.pt')

    # Train the model
    print("Starting training on TVD dataset...")
    results = model.train(
        data='data/tvd/data.yaml',
        epochs=50,                  # Number of training epochs
        imgsz=640,                  # Image size
        batch=16,                   # Batch size (reduce to 8 or 4 if GPU runs out of memory)
        device=device,              # Automatically use GPU if available
        project='runs/tvd',         # Directory to save results
        name='yolov11s_run1',       # Name of this training run
        exist_ok=True,              # Overwrite existing run with same name
        patience=10                 # Early stopping if no improvement for 10 epochs
    )

    print("\nTraining complete!")
    print("Your trained weights are saved at: runs/tvd/yolov11s_run1/weights/best.pt")

if __name__ == '__main__':
    main()
