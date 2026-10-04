from ultralytics import YOLO

def main():
    print("Loading AI VisionGuard model for Evaluation...")
    # Load the base model we have been using
    model = YOLO("yolov8n.pt") 
    
    print("Validating the model against the standard dataset...")
    # This automatically downloads a tiny 8-image validation set for quick testing
    # Without any arguments, dataset and settings are remembered from training, but we pass data explicitly here
    metrics = model.val(data="coco8.yaml") 
    
    print("\n📊 --- CORE EVALUATION METRICS ---")
    # Accessing the specific metric properties from the validation results
    print(f"mAP50-95 (Strict Accuracy): {metrics.box.map:.4f}")
    print(f"mAP50 (Lenient Accuracy):   {metrics.box.map50:.4f}")
    
    print("\n✅ Evaluation complete!")
    print("Check the newly created 'runs/detect/val' folder for your visual evaluation charts.")

if __name__ == "__main__":
    main()