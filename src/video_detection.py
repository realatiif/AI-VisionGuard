import cv2
import os
from ultralytics import YOLO

def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    
    input_video = "data/test_video.mp4"
    output_path = "outputs/result_test_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Error: Could not find {input_video}.")
        print("Please place a short video named 'test_video.mp4' inside the 'data' folder.")
        return

    print(f"Opening video {input_video}...")
    cap = cv2.VideoCapture(input_video)
    
    # Get original video properties to build the output video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing video frames... This might take a minute.")
    
    # Loop through the video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video
            
        # Run YOLO detection on the current frame
        results = model(frame, verbose=False)
        
        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the new video file
        out.write(annotated_frame)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    # Release the video objects from memory
    cap.release()
    out.release()
    print(f"✅ Video processing complete! Saved to {output_path}")

if __name__ == "__main__":
    main()