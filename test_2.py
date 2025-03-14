import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Set page config and add logo
st.set_page_config(page_title="Face Features Detection App", page_icon="😺", layout="wide")

# Display logo at the top
st.image("https://wallpapers.com/images/featured/cool-cat-1bdkaxbrpo86pxd3.jpg", width=150)

st.title("OpenCV Multi-Feature Detection App")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Feature Detection", "Person Search", "About"])

@st.cache_resource
def load_models():
    # Cargar varios clasificadores en cascada
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return {
        'face': face_cascade,
        'eye': eye_cascade,
        'smile': smile_cascade
    }

def detect_features(image, cascades, selected_features):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_features = {}
    
    for feature in selected_features:
        if feature == 'face':
            faces = cascades['face'].detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                detected_features['face'] = faces
            
                # Detectar ojos y sonrisas dentro de cada rostro si están seleccionados
                if 'eye' in selected_features or 'smile' in selected_features:
                    all_eyes = []
                    all_smiles = []
                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y+h, x:x+w]
                        
                        if 'eye' in selected_features:
                            eyes = cascades['eye'].detectMultiScale(roi_gray)
                            if len(eyes) > 0:
                                all_eyes.extend([(ex+x, ey+y, ew, eh) for (ex, ey, ew, eh) in eyes])
                        
                        if 'smile' in selected_features:
                            smiles = cascades['smile'].detectMultiScale(roi_gray, 1.7, 20)
                            if len(smiles) > 0:
                                all_smiles.extend([(sx+x, sy+y, sw, sh) for (sx, sy, sw, sh) in smiles])
                    
                    if all_eyes:
                        detected_features['eye'] = np.array(all_eyes)
                    if all_smiles:
                        detected_features['smile'] = np.array(all_smiles)
    
    return detected_features

def draw_features(image, features):
    colors = {
        'face': (255, 0, 0),    # Azul
        'eye': (0, 255, 0),     # Verde
        'smile': (0, 0, 255)    # Rojo
    }
    
    img_copy = image.copy()
    for feature_type, regions in features.items():
        if len(regions) > 0:  # Verificar que hay regiones para dibujar
            for region in regions:
                x, y, w, h = region
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), colors[feature_type], 2)
    
    return img_copy

def compare_faces(face1, face2):
    # Asegurar que las imágenes están en el mismo tamaño
    face1 = cv2.resize(face1, (128, 128))
    face2 = cv2.resize(face2, (128, 128))
    
    # Convertir a escala de grises
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    
    # Calcular la diferencia usando MSE
    difference = np.sum((face1_gray.astype("float") - face2_gray.astype("float")) ** 2)
    difference /= float(face1_gray.shape[0] * face1_gray.shape[1])
    
    return difference

if menu == "Feature Detection":
    st.sidebar.subheader("Seleccionar Características")
    detect_face = st.sidebar.checkbox("Detectar Rostros", True)
    detect_eyes = st.sidebar.checkbox("Detectar Ojos")
    detect_smile = st.sidebar.checkbox("Detectar Sonrisas")
    
    selected_features = []
    if detect_face: selected_features.append('face')
    if detect_eyes: selected_features.append('eye')
    if detect_smile: selected_features.append('smile')
    
    uploaded_file = st.file_uploader("Cargar imagen", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Cargar modelos
            cascades = load_models()
            
            # Procesar imagen
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Detectar características
                features = detect_features(image, cascades, selected_features)
                
                if features:
                    # Dibujar resultados
                    result_image = draw_features(image, features)
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    col1.image(image, channels="BGR", caption="Imagen Original")
                    col2.image(result_image, channels="BGR", caption="Detección de Características")
                    
                    # Estadísticas
                    st.subheader("Estadísticas de Detección")
                    for feature in selected_features:
                        if feature in features:
                            st.write(f"- {feature.title()} detectados: {len(features[feature])}")
                else:
                    st.warning("No se detectaron características en la imagen.")
            else:
                st.error("Error al cargar la imagen. Por favor, intenta con otra imagen.")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

elif menu == "Person Search":
    st.subheader("Búsqueda de Personas")
    
    # Cargar imagen de referencia
    st.sidebar.subheader("Imagen de Referencia")
    reference_file = st.sidebar.file_uploader("Cargar imagen de referencia", type=['jpg', 'jpeg', 'png'])
    
    # Cargar imágenes para buscar
    st.subheader("Imágenes para Buscar")
    search_files = st.file_uploader("Cargar imágenes para buscar", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if reference_file is not None and search_files:
        try:
            # Cargar imagen de referencia
            ref_bytes = np.asarray(bytearray(reference_file.read()), dtype=np.uint8)
            ref_image = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
            
            if ref_image is not None:
                # Extraer características de la imagen de referencia
                cascades = load_models()
                ref_features = detect_features(ref_image, cascades, ['face'])
                
                if 'face' in ref_features and len(ref_features['face']) > 0:
                    x, y, w, h = ref_features['face'][0]
                    ref_face = ref_image[y:y+h, x:x+w]
                    
                    st.sidebar.image(ref_face, channels="BGR", caption="Rostro de Referencia")
                    
                    # Buscar en las imágenes cargadas
                    st.subheader("Resultados de la Búsqueda")
                    for idx, search_file in enumerate(search_files):
                        search_bytes = np.asarray(bytearray(search_file.read()), dtype=np.uint8)
                        search_image = cv2.imdecode(search_bytes, cv2.IMREAD_COLOR)
                        
                        if search_image is not None:
                            search_features = detect_features(search_image, cascades, ['face'])
                            
                            if 'face' in search_features and len(search_features['face']) > 0:
                                matches = []
                                for face_coords in search_features['face']:
                                    x, y, w, h = face_coords
                                    face = search_image[y:y+h, x:x+w]
                                    try:
                                        face = cv2.resize(face, (ref_face.shape[1], ref_face.shape[0]))
                                        similarity = compare_faces(ref_face, face)
                                        matches.append((similarity, face_coords))
                                    except Exception as e:
                                        st.warning(f"Error al comparar rostros en imagen {idx + 1}: {str(e)}")
                                
                                if matches:
                                    best_match = min(matches, key=lambda x: x[0])
                                    similarity_score = best_match[0]
                                    
                                    col1, col2 = st.columns(2)
                                    col1.image(search_image, channels="BGR", caption=f"Imagen {idx + 1}")
                                    col2.write(f"Similitud: {max(0, min(100, similarity_score*100)):.2f}%")
                            else:
                                st.warning(f"No se detectaron rostros en la imagen {idx + 1}")
                        else:
                            st.error(f"Error al cargar la imagen {idx + 1}")
                else:
                    st.error("No se detectó ningún rostro en la imagen de referencia")
            else:
                st.error("Error al cargar la imagen de referencia")
        except Exception as e:
            st.error(f"Error en el proceso de búsqueda: {str(e)}")

elif menu == "About":
    st.subheader("Acerca de la Aplicación")
    st.write("""
    Esta es una aplicación mejorada de detección facial que incluye:
    
    **Características:**
    - Detección múltiple de características faciales (rostros, ojos, sonrisas)
    - Búsqueda de personas específicas en un conjunto de imágenes
    - Interfaz interactiva para selección de características
    - Estadísticas de detección
    
    **Tecnologías:**
    - OpenCV (Haar Cascades)
    - Streamlit
    - NumPy
    - PIL
    """)