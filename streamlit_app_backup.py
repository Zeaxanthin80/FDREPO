import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Set page config and add logo
st.set_page_config(page_title="Face Detection App", page_icon="😺", layout="wide")

# Display logo at the top
st.image("https://wallpapers.com/images/featured/cool-cat-1bdkaxbrpo86pxd3.jpg", width=150)

st.title("OpenCV Deep Learning Based Face Detection")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Face Detection", "About"])

if menu == "Face Detection":
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    # Function for Detecting face and annotating with rectangles
    def detectFaceOpenCVDnn(net, frame, conf_threshold=0.5):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]

        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False
        )
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(
                    frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0),
                    int(round(frameHeight / 150)), 8
                )
        return frameOpencvDnn, bboxes

    # Load DNN model
    @st.cache_resource
    def load_model():
        modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        return net

    def get_image_download_link(img, filename, text):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
        return href

    net = load_model()

    # Tabs for UI organization
    tab1, tab2 = st.tabs(["📸 Upload Image", "⚙️ Settings"])

    with tab1:
        if uploaded_file is not None:
            raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

            placeholders = st.columns(2)
            placeholders[0].image(opencv_image, channels='BGR', caption="Input Image")

            # Detect faces
            out_image, _ = detectFaceOpenCVDnn(net, opencv_image)

            placeholders[1].image(out_image, channels='BGR', caption="Output Image")

            # Convert OpenCV image to PIL for download
            out_image = Image.fromarray(out_image[:, :, ::-1])
            st.markdown(get_image_download_link(out_image, "face_output.jpg", '📥 Download Output Image'),
                        unsafe_allow_html=True)

    with tab2:
        conf_threshold = st.slider("Set Confidence Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        st.write(f"Current confidence threshold: **{conf_threshold}**")

elif menu == "About":
    st.subheader("About This App")
    st.write("""
    This is a **Deep Learning-based Face Detection App** using OpenCV's pre-trained SSD model.
    It allows you to upload an image, detects faces, and provides a downloadable output.
    
    **Features:**
    - Uses OpenCV's Deep Learning module for face detection.
    - Adjustable confidence threshold via a slider.
    - Downloadable processed images.
    - Interactive UI with tabs and sidebar navigation.

    **Built with:**  
    - Streamlit  
    - OpenCV  
    - PIL  
    """)

    st.markdown("👨‍💻 Developed by [Your Name]")  # Replace with your name

