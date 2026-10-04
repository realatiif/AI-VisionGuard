import cv2
import os
import numpy as np
from ultralytics import YOLO

def main():
    print("Loading YOLO model for Safety Analytics...")
    model = YOLO("yolov8n.pt")
    
    input_video = "data/test_video.mp4"
    output_path = "outputs/result_safety_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Error: Could not find {input_video}.")
        return

    print("Initializing video streams...")
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define a virtual Restricted Zone (a polygon near the bottom/center of the frame)
    restricted_zone = np.array([
        [int(width*0.2), int(height*0.7)], 
        [int(width*0.8), int(height*0.7)], 
        [int(width*0.9), int(height*0.95)], 
        [int(width*0.1), int(height*0.95)]
    ], np.int32)
    
    print("Processing video and calculating ML risk scores...")
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Run YOLO tracking (we do not use plot() here because we will draw custom boxes)
        results = model.track(frame, persist=True, verbose=False)
        result = results[0]
        
        # 2. Draw the restricted zone on the frame
        cv2.polylines(frame, [restricted_zone], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "RESTRICTED ZONE", (int(width*0.22), int(height*0.75)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        people_count = 0
        violations = 0
        
        # 3. Analyze detections mathematically
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            
            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                # Class 0 in COCO dataset is 'person'
                if int(cls_id) == 0:
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate the center bottom of the person's bounding box (their feet)
                    cx, cy = (x1 + x2) // 2, y2
                    
                    # Check if the person's coordinates fall inside our restricted zone polygon
                    in_zone = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0
                    
                    if in_zone:
                        violations += 1
                        # Draw a RED box and warning label for violations
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                        cv2.putText(frame, f"WARNING ID:{int(track_id)}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Draw a GREEN box for safe people
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        cv2.putText(frame, f"ID:{int(track_id)}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
        # 4. Calculate Risk Score Engine
        base_risk = 5
        risk_score = min(100, base_risk + (people_count * 2) + (violations * 40))
        risk_level = "CRITICAL" if risk_score > 75 else "HIGH" if risk_score > 50 else "LOW"
        risk_color = (0, 0, 255) if risk_level in ["CRITICAL", "HIGH"] else (0, 255, 0)
        
        # 5. Draw the Analytics Dashboard on the video frame
        # We use a semi-transparent black rectangle as a background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"AI VISIONGUARD ANALYTICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People Tracked: {people_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Zone Violations: {violations}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if violations > 0 else (255, 255, 255), 1)
        cv2.putText(frame, f"System Risk: {risk_level} ({risk_score}%)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Analyzed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"✅ Safety analysis complete! Saved to {output_path}")

if __name__ == "__main__":
    main()