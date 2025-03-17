# Face Features Detection App

Welcome to the **Face Features Detection App**! This application is designed to detect facial features using OpenCV and Streamlit, providing a user-friendly interface for image processing and analysis.

## Project Evolution 🚀

### Initial Version: test_1.py
- **Objective**: Basic face detection using OpenCV's DNN module.
- **Features**:
  - Single image upload for face detection.
  - Utilized a pre-trained SSD model for face detection.
  - Basic UI with a sidebar for navigation and a simple face comparison feature.

### Intermediate Version: test_2.py
- **Enhancements**:
  - Added multi-feature detection (faces, eyes, smiles) using Haar Cascades.
  - Introduced a more interactive UI with options to select features for detection.
  - Implemented a person search feature to find similar faces across multiple images.
  - Improved performance with caching of models.

### Current Version: streamlit_app.py
- **Comprehensive Features**:
  - **Multi-Feature Detection**: Detects faces, eyes, and smiles with adjustable settings for detection scale and confidence.
  - **Advanced Person Search**: Allows searching for similar faces in a batch of images with various comparison methods.
  - **Detailed Statistics**: Provides visual statistics of detected features using Plotly.
  - **Export Options**: Users can download processed images and detection statistics.
  - **Enhanced UI**: Modern and responsive design with tabs for better organization and user experience.

## How to Use 🛠️
1. **Select Mode**: Choose between Feature Detection, Person Search, or About from the sidebar.
2. **Adjust Settings**: Configure detection settings and upload images.
3. **Process Images**: View results, statistics, and export options.
4. **Explore**: Use the About section to learn more about the app's features and technologies.

## Technologies Used 🧰
- **OpenCV**: For image processing and feature detection.
- **Streamlit**: To create an interactive web application.
- **NumPy**: For numerical operations and image manipulation.
- **PIL**: For image handling and processing.
- **Plotly**: For creating interactive visualizations.

## Support & Contributions 🤝
For any issues or contributions, please visit our [GitHub repository](#). We welcome feedback and suggestions to improve the app.

---

Thank you for using the Face Features Detection App! We hope it serves your needs in facial feature detection and analysis. 😊

