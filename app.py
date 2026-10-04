import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import requests
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="AI VisionGuard", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🛡️ AI VisionGuard")
st.markdown("**Intelligent Real-Time Safety & Behavior Analysis System**")
st.markdown("---")

st.sidebar.header("System Controls")
hf_token = st.sidebar.text_input("Hugging Face API Token", type="password", help="Enter your token to enable live GenAI reporting.")
confidence = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
uploaded_video = st.sidebar.file_uploader("Upload Surveillance Video", type=['mp4', 'mov', 'avi'])

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Analytics")
    metric_people = st.empty()
    metric_violations = st.empty()
    metric_risk = st.empty()
    
    st.markdown("---")
    st.subheader("📝 GenAI Reporting")
    report_placeholder = st.empty()

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    restricted_zone = np.array([
        [int(width*0.2), int(height*0.7)], 
        [int(width*0.8), int(height*0.7)], 
        [int(width*0.9), int(height*0.95)], 
        [int(width*0.1), int(height*0.95)]
    ], np.int32)
    
    max_people_seen = 0
    unique_violators = set()
    peak_risk = 0
    
    analytics_data = []
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        results = model.track(frame, persist=True, conf=confidence, verbose=False)
        result = results[0]
        
        cv2.polylines(frame, [restricted_zone], isClosed=True, color=(0, 0, 255), thickness=2)
        
        current_people = 0
        current_violations = 0
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            
            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                # Class 0 is 'person'. To track vehicles too, change to: if int(cls_id) in [0, 2, 3, 5, 7]:
                if int(cls_id) == 0:  
                    current_people += 1
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, y2
                    
                    in_zone = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0
                    
                    if in_zone:
                        current_violations += 1
                        unique_violators.add(int(track_id))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        max_people_seen = max(max_people_seen, current_people)
        
        base_risk = 5
        risk_score = min(100, base_risk + (current_people * 2) + (current_violations * 40))
        peak_risk = max(peak_risk, risk_score)
        
        risk_level = "CRITICAL 🔴" if risk_score > 75 else "HIGH 🟠" if risk_score > 50 else "LOW 🟢"
        
        analytics_data.append({
            "Frame": frame_count,
            "People Tracked": current_people,
            "System Risk": risk_score
        })
        
        metric_people.metric("Active People", current_people)
        metric_violations.metric("Active Violations", current_violations)
        metric_risk.metric("Current Risk", risk_level, f"{risk_score}%")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()
    
    # ---------------------------------------------
    # POST-PROCESSING: Analytics & LIVE GenAI Reporting
    # ---------------------------------------------
    
    total_violations = len(unique_violators)
    final_risk_category = "CRITICAL" if peak_risk > 75 else "HIGH" if peak_risk > 50 else "LOW"
    
    st.markdown("---")
    st.subheader("📈 Post-Analysis Data Visualization")
    
    df_analytics = pd.DataFrame(analytics_data)
    df_analytics.set_index("Frame", inplace=True)
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Crowd Density Over Time**")
        st.area_chart(df_analytics["People Tracked"], color="#1f77b4")
        
    with chart_col2:
        st.markdown("**System Risk Fluctuations**")
        st.line_chart(df_analytics["System Risk"], color="#ff7f0e")
    
    # ---------------------------------------------
    # LIVE HUGGING FACE API CALL WITH GRACEFUL FALLBACK
    # ---------------------------------------------
    st.markdown("### 📄 AI Incident Report")
    
    if hf_token:
        with st.spinner("🧠 Generating live AI incident report via Hugging Face..."):
            # Serverless Inference API Endpoint
            API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
            headers = {"Authorization": f"Bearer {hf_token}"}
            
            prompt = f"""<|user|>
            You are a professional AI security system. Write a concise, 3-paragraph incident report based on these live metrics:
            - Peak Crowd Density: {max_people_seen} people
            - Unique Zone Violations: {total_violations}
            - Peak System Risk Score: {peak_risk}% ({final_risk_category})
            
            Use these exact headings: **Executive Summary**, **Incident Analysis**, and **Recommended Action**. Do not include conversational filler.</s>
            <|assistant|>
            """
            
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 300, "temperature": 0.3, "return_full_text": False}
            }
            
            try:
                # 10-second timeout ensures the app does not freeze if the DNS lookup fails
                response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    generated_report = response.json()[0]["generated_text"].strip()
                    st.info(generated_report)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.warning("⚠️ Live API connection blocked by local network. Generating fallback report...")
                
                fallback_report = f"""
                **Executive Summary:**
                The AI VisionGuard system processed the footage and detected a peak crowd density of **{max_people_seen} individuals**. 
                
                **Incident Analysis:**
                During the monitoring period, **{total_violations} unique perimeter violations** occurred, escalating the system's peak risk score to **{peak_risk}% ({final_risk_category})**.
                
                **Recommended Action:**
                Review access controls for the restricted perimeter.
                
                *Report auto-generated by AI VisionGuard Fallback Engine*
                *System Officer: Muhammad Atif*
                """
                st.info(fallback_report)
    else:
        st.warning("⚠️ **Live GenAI Disabled.** Please enter your Hugging Face API token in the sidebar to generate a dynamic incident report.")

else:
    st.info("👈 Please upload a video from the sidebar to start the AI VisionGuard engine.")