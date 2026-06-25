import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
import tempfile

# Page config
st.set_page_config(
    page_title="Intelligent Face Detection & Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #ffc107; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    from tensorflow.keras.models import load_model

    YOLO_MODEL_PATH = "../models/WIDER_FACE/runs/detect/face_detection/weights/best.pt"
    EMOTION_MODEL_PATH = "../models/FER2013/best_fer2013_model.keras"
    
    face_model = YOLO(YOLO_MODEL_PATH)
    expr_model = load_model(EMOTION_MODEL_PATH)
    return face_model, expr_model

try:
    face_detector, expression_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SENTIMENT_MAP = {
    'angry': 'Negative', 'disgust': 'Negative', 'fear': 'Negative',
    'happy': 'Positive', 'neutral': 'Neutral', 'sad': 'Negative',
    'surprise': 'Positive'
}
EMOTION_COLORS = {
    'angry': '#FF6B6B', 'disgust': '#4ECDC4', 'fear': '#45B7D1',
    'happy': '#96CEB4', 'neutral': '#FFEAA7', 'sad': '#DDA0DD',
    'surprise': '#FFD93D'
}

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/face-id.png", width=80)
    st.title("Settings")
    
    if models_loaded:
        st.success("Models Loaded")
    else:
        st.error("Models Not Loaded")
    
    st.markdown("---")
    
    # Detection settings
    st.subheader("Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    max_faces = st.slider("Max Faces", 1, 50, 20)
    
    st.markdown("---")
    
    # Model info
    st.subheader("Model Info")
    st.info("""
    **Face Detection:** YOLOv8n  
    **Expression:** CNN (FER2013)  
    **Classes:** 7 Emotions  
    **Sentiment:** Positive/Negative/Neutral
    """)
    
    st.markdown("---")
    st.caption("Professional Training - Final Project")

# ============================================
# MAIN HEADER
# ============================================
st.markdown('<p class="main-header">Intelligent Face Detection & Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time face detection, expression recognition, and sentiment analysis</p>', unsafe_allow_html=True)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload Image", 
    "Real-Time Camera", 
    "Model Metrics", 
    "Summary & Export"
])

# ============================================
# TAB 1: UPLOAD IMAGE
# ============================================
with tab1:
    st.header("Upload Image Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Upload an image containing faces for analysis"
    )
    
    if uploaded_file and models_loaded:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Read image
        image = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        with st.spinner("Analyzing faces..."):
            start_time = time.time()
            
            results = face_detector(tmp_path, conf=conf_threshold, verbose=False)
            detections = results[0].boxes
            
            face_results = []
            annotated = image.copy()
            
            for i, box in enumerate(detections[:max_faces]):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                det_conf = float(box.conf[0])
                
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # Expression prediction
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = face_resized.reshape(1, 48, 48, 1) / 255.0
                
                pred = expression_model.predict(face_input, verbose=0)
                pred_idx = np.argmax(pred)
                expr_conf = np.max(pred) * 100
                
                emotion = EMOTIONS[pred_idx]
                sentiment = SENTIMENT_MAP[emotion]
                
                face_results.append({
                    'Face #': i + 1,
                    'Emotion': emotion,
                    'Emotion Confidence': f"{expr_conf:.1f}%",
                    'Sentiment': sentiment,
                    'Detection Confidence': f"{det_conf:.3f}",
                    'BBox': f"[{x1}, {y1}, {x2}, {y2}]"
                })
                
                # Draw
                color = (0, 255, 0) if sentiment == 'Positive' else \
                        (0, 0, 255) if sentiment == 'Negative' else (0, 255, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{emotion} | {sentiment} | {expr_conf:.0f}%"

                font_scale = 1
                thickness = 2

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 12),
                    (x1 + tw, y1),
                    color,
                    -1
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )
            
            inference_time = (time.time() - start_time) * 1000
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, width="stretch")
        
        with col2:
            st.subheader("Analyzed Image")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        
        # Metrics cards
        if face_results:
            st.subheader("Detection Summary")
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Faces Detected", len(face_results))
            with mcol2:
                avg_conf = np.mean([float(r['Emotion Confidence'].rstrip('%')) for r in face_results])
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with mcol3:
                st.metric("Inference Time", f"{inference_time:.0f}ms")
            with mcol4:
                dominant = Counter([r['Sentiment'] for r in face_results]).most_common(1)[0][0]
                st.metric("Dominant Mood", dominant)
            
            # Results table
            st.subheader("Detailed Results")
            df = pd.DataFrame(face_results)
            
            # Color code sentiments
            def color_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Negative':
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return 'background-color: #fff3cd; color: #856404'
            
            styled_df = df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Charts
            ccol1, ccol2 = st.columns(2)
            
            with ccol1:
                st.subheader("Emotion Distribution")
                emotion_counts = df['Emotion'].value_counts()
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    color=emotion_counts.index,
                    color_discrete_map=EMOTION_COLORS,
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with ccol2:
                st.subheader("Sentiment Distribution")
                sent_counts = df['Sentiment'].value_counts()
                colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'}
                fig = px.bar(
                    x=sent_counts.index,
                    y=sent_counts.values,
                    color=sent_counts.index,
                    color_discrete_map=colors,
                    labels={'x': 'Sentiment', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence bar chart
            st.subheader("Confidence per Face")
            conf_data = {
                'Face': [f"Face {r['Face #']}" for r in face_results],
                'Confidence': [float(r['Emotion Confidence'].rstrip('%')) for r in face_results],
                'Emotion': [r['Emotion'] for r in face_results]
            }
            fig = px.bar(
                conf_data,
                x='Face',
                y='Confidence',
                color='Emotion',
                color_discrete_map=EMOTION_COLORS,
                range_y=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Results (CSV)",
                csv,
                "face_analysis_results.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.warning("No faces detected. Try lowering the confidence threshold.")
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

# ============================================
# TAB 2: REAL-TIME CAMERA
# ============================================
with tab2:
    st.header("Real-Time Face Detection & Analysis")
    
    
    # Camera settings
    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        show_bbox = st.checkbox("Show Bounding Boxes", value=True)
    with cam_col2:
        show_labels = st.checkbox("Show Labels", value=True)
    
    # We use OpenCV for camera since Streamlit's native camera_input is for single capture
    run_camera = st.button("Start Camera", use_container_width=True)
    stop_camera = st.button("Stop Camera", use_container_width=True)
    
    FRAME_WINDOW = st.image([])
    
    if run_camera and models_loaded:
        st.success("Camera started! Press 'Stop' to end.")
        
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Detect faces
            results = face_detector(frame, conf=conf_threshold, verbose=False)
            detections = results[0].boxes
            
            # Process each face
            for box in detections[:max_faces]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # Expression
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = face_resized.reshape(1, 48, 48, 1) / 255.0
                
                pred = expression_model.predict(face_input, verbose=0)
                pred_idx = np.argmax(pred)
                confidence = np.max(pred) * 100
                
                emotion = EMOTIONS[pred_idx]
                sentiment = SENTIMENT_MAP[emotion]
                
                # Draw
                if show_bbox:
                    color = (0, 255, 0) if sentiment == 'Positive' else \
                            (0, 0, 255) if sentiment == 'Negative' else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if show_labels:
                    label = f"{emotion} | {sentiment}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            # Check stop button (Streamlit limitation: need external stop)
            if stop_camera:
                break
        
        cap.release()
        st.info("Camera stopped")
    
    elif not models_loaded:
        st.error("Models not loaded. Cannot start camera.")

# ============================================
# TAB 3: MODEL METRICS
# ============================================
with tab3:
    st.header("Model Performance Metrics")

    mcol1, mcol2 = st.columns(2)

    # ============================================
    # YOLOv8 FACE DETECTION
    # ============================================
    with mcol1:
        st.subheader("Face Detection (YOLOv8)")

        det_metrics_path = Path("../results/face_detection_validation.csv")

        if det_metrics_path.exists():
            det_df = pd.read_csv(det_metrics_path)

            st.dataframe(det_df, use_container_width=True)

        else:
            st.warning("results/face_detection_validation.csv not found")


        # Optional image
        performance_img = Path("img/performance_charts.png")

        if performance_img.exists():
            st.image(
                str(performance_img),
                caption="YOLOv8 Detection Performance Charts",
                use_container_width=True
            )

    with mcol2:
        st.subheader("Expression Recognition (CNN)")

        expr_metrics_path = Path(
            "../results/fer2013_overall_metrics.csv"
        )

        if expr_metrics_path.exists():

            expr_df = pd.read_csv(expr_metrics_path)

            st.dataframe(expr_df, use_container_width=True)

        else:
            st.warning(
                "results/fer2013_overall_metrics.csv not found"
            )

    
    # Confusion matrix placeholder
    st.subheader("Expression Confusion Matrix (Normalized)")

    cm_csv_path = Path("../results/fer2013_confusion_matrix_normalized.csv")

    if cm_csv_path.exists():
        cm_df = pd.read_csv(cm_csv_path, index_col=0)
        
        # Convert to actual percentages (multiply by 100)
        cm_percent = (cm_df * 100).round(1)
        
        fig = px.imshow(
            cm_percent.values,
            labels=dict(x="Predicted", y="True", color="Accuracy %"),
            x=cm_percent.columns,
            y=cm_percent.index,
            color_continuous_scale='Blues',
            text_auto='.1f',  # Show 1 decimal
            aspect="auto",
            range_color=[0, 100]
        )
        
        # Fix: Use proper text template
        fig.update_traces(
            texttemplate='%{z:.1f}',
            textfont_size=11
        )
        
        fig.update_layout(
            title='Normalized Confusion Matrix - Expression Recognition (%)',
            title_font_size=16,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info("⚠️ Confusion matrix not found.")

# ============================================
# TAB 4: SUMMARY & EXPORT
# ============================================
with tab4:
    st.header("Project Summary & Export")
    
    st.subheader("System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │              STREAMLIT WEB APPLICATION                  │
    ├─────────────────────────────────────────────────────────┤
    │  Upload Image  │  Real-Time Camera  │  Metrics  │  Info │
    ├─────────────────────────────────────────────────────────┤
    │  ┌─────────────┐    ┌─────────────────────────────────┐ │
    │  │ YOLOv8 Face │───→│      Face Crop (bbox)           │ │
    │  │  Detection  │    └─────────────────────────────────┘ │
    │  └─────────────┘                    │                   │
    │         │                           ▼                   │
    │         │              ┌─────────────────────┐          │
    │         │              │  Grayscale + Resize │          │
    │         │              │     (48 x 48)       │          │
    │         │              └─────────────────────┘          │
    │         │                           │                   │
    │         │                           ▼                   │
    │         │              ┌─────────────────────┐          │
    │         │              │   CNN Expression    │          │
    │         └─────────────→│    Recognition      │          │
    │                        └─────────────────────┘          │
    │                                     │                   │
    │                                     ▼                   │
    │                        ┌─────────────────────┐          │
    │                        │  Sentiment Mapping  │          │
    │                        │  (Positive/Negative/│          │
    │                        │       Neutral)      │          │
    │                        └─────────────────────┘          │
    └─────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.subheader("Features")
    features = {
        "Face Detection": "YOLOv8n trained on WIDER Face dataset",
        "Expression Recognition": "CNN trained on FER2013 (7 emotions)",
        "Sentiment Analysis": "Automatic mapping to Positive/Negative/Neutral",
        "Real-Time Processing": "Webcam integration for live analysis",
        "Interactive Visualizations": "Charts, tables, and confidence metrics",
        "Export Results": "Download CSV with full analysis"
    }
    
    for feature, desc in features.items():
        st.markdown(f"**{feature}** — {desc}")
    
    st.subheader("Model Specifications")
    
    spec_col1, spec_col2 = st.columns(2)
    
    with spec_col1:
        st.markdown("""
        **Face Detection Model**
        - Architecture: YOLOv8n (Nano)
        - Dataset: WIDER Face
        - Input Size: 640x640
        - Classes: 1 (face)
        - Output: Bounding boxes + confidence
        """)
    
    with spec_col2:
        st.markdown("""
        **Expression Model**
        - Architecture: CNN (4 Conv blocks)
        - Dataset: FER2013
        - Input Size: 48x48 grayscale
        - Classes: 7 emotions
        - Output: Probabilities + sentiment
        """)
    
    st.subheader("Export Options")
    
    if st.button("Generate Full Report", use_container_width=True):
        st.success("Report generation feature coming soon!")
    
    if st.button("Export All Metrics", use_container_width=True):
        st.success("Metrics export feature coming soon!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Professional Training Final Project | Intelligent Face Detection & Analysis System</p>
    <p>Built with Streamlit, YOLOv8, TensorFlow/Keras</p>
</div>
""", unsafe_allow_html=True)