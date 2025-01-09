# Signature Verification System
SigVerify is a signature verification application leveraging TensorFlow-based CNN models for automated matching and Tkinter for a user-friendly GUI. It combines OpenCV for real-time signature capture and SSIM for structural similarity analysis, ensuring robust and accurate verification.

## Tools Required

### Hardware Requirements
- **Processor**: Intel Core i5 or higher.
- **RAM**: 8 GB or more.
- **Storage**: Minimum of 1 GB free space.
- **Webcam**: Optional, for capturing signatures directly.

### Software Requirements
- **Operating System**: Windows 10, macOS, or Linux.
- **Python**: Version 3.x installed on your system.
- **Python Libraries**:
  - TensorFlow
  - OpenCV
  - Pillow
  - NumPy
  - Scikit-image

---

## Setup and Installation

### 1. Install Python
- Download Python from the official website: [https://www.python.org/](https://www.python.org/).
- Ensure you add Python to your system PATH during installation.

### 2. Extract Project Files
- Extract the zipped project file from the provided CD to a folder on your system.
- Verify that the extracted folder contains:
  - Source files (main.py, ui.py, etc.).
  - Data folder containing the dataset for training the model. You can modify it according to your data available and divide them into real or forged subfolders.
  - A models folder containing `model_cnn.h5`.
  - Any media files used (e.g., videos, images).

### 3. Install Required Libraries
- Open a terminal/command prompt and navigate to the project folder.
- Install the required dependencies:

  If `requirements.txt` is missing, install libraries manually:
  
  ```bash
  pip install tensorflow opencv-python pillow numpy scikit-image

### 4. Execution of Software
- Go through User Manual.docx file for step by step guidance


  
