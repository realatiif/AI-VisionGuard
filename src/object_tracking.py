import cv2
import os
from ultralytics import YOLO

def main():
    print("Loading YOLO model for Tracking...")
    model = YOLO("yolov8n.pt")
    
    input_video = "data/test_video.mp4"
    output_path = "outputs/result_tracking_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Error: Could not find {input_video}.")
        return

    print(f"Opening video {input_video} for tracking...")
    cap = cv2.VideoCapture(input_video)
    
    # Get original video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing video frames with object tracking...")
    
    # Loop through the video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Run YOLO TRACKING on the current frame
        # persist=True is crucial: it tells the model to remember IDs from the previous frames
        results = model.track(frame, persist=True, verbose=False)
        
        # Draw bounding boxes AND tracking IDs on the frame
        annotated_frame = results[0].plot()
        
        # Write the annotated frame
        out.write(annotated_frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Tracked {frame_count} frames...")
            
    # Release memory
    cap.release()
    out.release()
    print(f"✅ Tracking complete! Saved to {output_path}")

if __name__ == "__main__":
    main()