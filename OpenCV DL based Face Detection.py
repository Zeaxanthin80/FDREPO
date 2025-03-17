import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os

# Create application title and file uploader widgets
st.title("OpenCV Deep Learning based Face Detection")
st.sidebar.title("Settings")
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])

if input_type == "Image":
    file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
else:
    file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)

    # Get Detections.
    detections = net.forward()
    return detections
    


# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# Function to load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Function to process video
def process_video(video_path, net, conf_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a temporary file for the processed video
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process each frame
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress bar
        current_frame += 1
        progress_bar.progress(current_frame / frame_count)
        
        # Detect faces in the frame
        detections = detectFaceOpenCVDnn(net, frame)
        processed_frame, _ = process_detections(frame, detections, conf_threshold)
        
        # Write the frame
        out.write(processed_frame)
    
    # Release resources
    cap.release()
    out.release()
    progress_bar.empty()
    
    return temp_output_path

net = load_model()

if file_buffer is not None:
    if input_type == "Image":
        # Read the file and convert it to opencv Image
        raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        # Create placeholders to display input and output images
        placeholders = st.columns(2)
        placeholders[0].image(image, channels='BGR')
        placeholders[0].text("Input Image")

        # Create a Slider and get the threshold from the slider
        conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

        # Call the face detection model to detect faces in the image
        detections = detectFaceOpenCVDnn(net, image)
        out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold)

        # Display Detected faces
        placeholders[1].image(out_image, channels='BGR')
        placeholders[1].text("Output Image")

        # Convert opencv image to PIL
        out_image = Image.fromarray(out_image[:, :, ::-1])
        st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
                    unsafe_allow_html=True)
    
    else:  # Video processing
        # Save uploaded video to temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "input_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(file_buffer.read())
            
        # Display original video
        st.video(temp_path)
        
        # Process video when button is clicked
        if st.button("Process Video"):
            conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)
            st.text("Processing video... Please wait.")
            
            # Process the video
            output_path = process_video(temp_path, net, conf_threshold)
            
            # Display processed video
            st.text("Processing complete! Here's the result:")
            st.video(output_path)
            
            # Provide download link
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button(
                label="Download processed video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
            
            # Clean up temporary files
            os.remove(temp_path)
            os.remove(output_path)
            os.rmdir(temp_dir)
