# AnalyCell - Automated Cell Nuclei Counter

An AI-powered web application for automated cell nuclei counting in microscopy images and binary mask generation using deep learning.

## 🔬 Overview

AnalyCell is a Flask-based web application that uses a U-Net deep learning model to automatically detect and count cell nuclei in microscopy images. The system processes images in seconds and provides three types of outputs: automatic nuclei counting, precise binary masks, and confidence heatmaps.

**Note: This project was developed as a learning exercise** to understand deep learning techniques applied to biomedical image analysis and to master the complete pipeline from training to deployment. The next step will be to train the model with real microscopy images for actual laboratory use.

## 🚀 Features

- **Automated Nuclei Detection**: Uses U-Net architecture to identify cell nuclei
- **Multiple Output Types**:
  - Automatic cell count
  - Binary segmentation masks
  - Confidence heatmaps showing model predictions
- **Web Interface**: User-friendly Flask application with file upload
- **Image Format Support**: TIFF, PNG, JPG, JPEG
- **Real-time Processing**: Results generated in seconds
- **Advanced Image Processing**: Includes morphological filtering, watershed segmentation, and geometric filtering

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, U-Net architecture
- **Image Processing**: OpenCV, scikit-image, PIL
- **Frontend**: HTML, Jinja2 templates
- **Deployment**: Gunicorn (production ready)

## 📁 Project Structure

```
analycell/
├── app.py              # Flask web application
├── model.py            # U-Net model architecture
├── predict.py          # Inference and image processing
├── train.py            # Training script
├── dataset.py          # Custom dataset class
├── requirements.txt    # Python dependencies
├── Procfile           # Deployment configuration
├── cell_counter_nuclear.pth  # Trained model weights
└── templates/         # HTML templates
```

## 🎯 How It Works

1. **Image Upload**: User uploads microscopy image through web interface
2. **Preprocessing**: Image is converted to grayscale and normalized
3. **AI Inference**: U-Net model generates probability heatmap
4. **Post-processing**: 
   - Gaussian filtering and morphological operations
   - Region filtering by area, circularity, and aspect ratio
   - Peak detection and watershed segmentation
5. **Output Generation**: Three visualizations are created and displayed

## 📊 Training Data

Used the image set [BBBC005v1](https://bbbc.broadinstitute.org/bbbc/BBBC005) from the Broad Bioimage Benchmark Collection [[Ljosa et al., Nature Methods, 2012](http://dx.doi.org/10.1038/nmeth.2083)]. The dataset contains simulated high-content screening (HCS) images generated with the [SIMCEP](http://www.cs.tut.fi/sgn/csb/simcep/tool.html) platform for fluorescent cell population images.

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Flask and other dependencies (see requirements.txt)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/nereasaga/analycell.git
cd analycell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5001`

### Using the Application
1. Upload a microscopy image (TIFF, PNG, JPG, JPEG)
2. Click "Process Image"
3. View results:
   - Cell count number
   - Original image with detected contours
   - Binary segmentation mask
   - Confidence heatmap

## 🏗️ Model Architecture

The system uses a custom U-Net implementation with:
- **Encoder**: 2 convolutional blocks with max pooling
- **Bottleneck**: 1 convolutional block
- **Decoder**: 2 transpose convolutional blocks with skip connections
- **Output**: Single channel probability map

Training uses distance transform maps as targets for better boundary detection.

## 🔄 Future Development

- **Real Image Training**: Plan to retrain with actual microscopy images from laboratory settings
- **Enhanced Features**: Multi-class cell type detection
- **Performance Optimization**: GPU acceleration for batch processing

## 🤝 Contributing

This project is primarily for educational purposes. Feedback and suggestions are welcome!


***

**Disclaimer**: This is a learning project using simulated data. For production use in research or clinical settings, the model should be retrained and validated with real microscopy images and appropriate ground truth data.
