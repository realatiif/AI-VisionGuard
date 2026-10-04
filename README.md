# 🛡️ AI VisionGuard
**Intelligent Real-Time Safety & Behavior Analysis System**

AI VisionGuard is an end-to-end computer vision and machine learning pipeline designed to monitor video feeds, track human movement, and calculate real-time safety risks based on spatial violations.

## 🧠 System Architecture
1. **Deep Learning Vision:** Utilizes YOLOv8 for high-speed object detection and persistent tracking (BoT-SORT).
2. **Spatial Analytics:** OpenCV calculates bounding box coordinates against a defined virtual restricted perimeter.
3. **Risk Engine:** A custom algorithm calculates dynamic risk scores based on crowd density and zone violations.
4. **Data Visualization:** Pandas tracks frame-by-frame metrics to generate time-series anomaly charts.
5. **Interactive UI:** Deployed as a Streamlit web application for real-time video tensor processing.

## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/yourusername/AI-VisionGuard.git](https://github.com/yourusername/AI-VisionGuard.git)
cd AI-VisionGuard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Dashboard
streamlit run app.py