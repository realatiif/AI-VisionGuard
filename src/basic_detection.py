from ultralytics import YOLO
import os

def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt") 

    input_image = "data/test_image.jpg"
    output_dir = "outputs"

    if not os.path.exists(input_image):
        print(f"❌ Error: Could not find {input_image}.")
        return

    print(f"Running AI detection on {input_image}...")

    results = model(input_image)
    result = results[0]

    output_path = os.path.join(output_dir, "result_test_image.jpg")
    result.save(filename=output_path)

    print(f"✅ Success! Check the '{output_dir}' folder to see the annotated image.")

if __name__ == "__main__":
    main()