
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

def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (128, 128))
    face2 = cv2.resize(face2, (128, 128))
    difference = np.sum((face1.astype("float") - face2.astype("float")) ** 2)
    difference /= float(face1.shape[0] * face1.shape[1])
    return difference

def extract_face(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

net = load_model()

# Tabs for UI organization
tab1, tab2 = st.tabs(["📸 Upload Image", "⚙️ Settings"])

if menu == "Face Detection":
    # Modificar el file uploader para permitir múltiples archivos
    uploaded_files = st.file_uploader("Sube tus imágenes", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    with tab1:
        if uploaded_files and len(uploaded_files) > 0:
            # Crear contenedores para las imágenes
            cols = st.columns(len(uploaded_files))
            processed_images = []
            detected_faces = []

            # Procesar cada imagen
            for idx, uploaded_file in enumerate(uploaded_files):
                raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                # Detectar rostros
                out_image, bboxes = detectFaceOpenCVDnn(net, opencv_image)
                
                # Mostrar imagen en su columna
                cols[idx].image(out_image, channels='BGR', caption=f"Imagen {idx + 1}")
                
                # Extraer y guardar rostros detectados
                faces = [extract_face(opencv_image, bbox) for bbox in bboxes]
                detected_faces.extend(faces)
                processed_images.append(out_image)

            # Comparar rostros si se detectaron
            if len(detected_faces) > 1:
                st.subheader("Comparación de Rostros")
                comparison_cols = st.columns(2)
                
                # Selector para elegir rostros a comparar
                face1_idx = comparison_cols[0].selectbox("Seleccionar primer rostro", 
                                                       range(len(detected_faces)), 
                                                       format_func=lambda x: f"Rostro {x+1}")
                face2_idx = comparison_cols[1].selectbox("Seleccionar segundo rostro", 
                                                       range(len(detected_faces)), 
                                                       format_func=lambda x: f"Rostro {x+1}")
                
                if st.button("Comparar Rostros"):
                    similarity = compare_faces(detected_faces[face1_idx], detected_faces[face2_idx])
                    st.write(f"Diferencia entre rostros: {similarity:.2f}")
                    st.write("Menor diferencia indica mayor similitud")

                    # Mostrar rostros seleccionados
                    comp_cols = st.columns(2)
                    comp_cols[0].image(detected_faces[face1_idx], channels='BGR', caption="Rostro 1")
                    comp_cols[1].image(detected_faces[face2_idx], channels='BGR', caption="Rostro 2")

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

    st.markdown("👨‍💻 Developed by [Your Name]")