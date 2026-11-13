# Speech Recognition Using Convolutional Neural Networks (CNN)

A deep learning project that recognises spoken words from audio recordings using a **1D Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

The project explores audio preprocessing, neural network design, model training, and performance evaluation for keyword spotting.

## Project Overview

Speech recognition enables computers to identify spoken language and convert audio into meaningful information. This project focuses on recognising individual spoken words from audio signals using a CNN.

The model is trained on recordings from the **Google Speech Commands dataset**. Audio files are preprocessed and converted into fixed-length numerical waveforms before being passed into the neural network for classification.

## Key Features

- Audio preprocessing and resampling
- Spoken-word classification using a 1D CNN
- Multi-class classification across 30 word categories
- Dropout regularisation to help reduce overfitting
- Early stopping during training
- Model evaluation using accuracy, precision, recall, and F1-score

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **SciPy**
- **Librosa**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Google Colab**

## Dataset

This project uses the **Google Speech Commands dataset**, which contains short audio recordings of spoken words.

The dataset contains approximately 65,000 one-second recordings covering 30 spoken-word categories.

The audio dataset is not included in this repository. The notebook uses audio files stored in Google Drive, so the dataset directory must be configured before running the complete workflow.

## Audio Preprocessing

Audio preprocessing prepares the recordings for the CNN.

The notebook performs the following steps:

1. Loads WAV audio files.
2. Resamples the recordings from 16 kHz to 8 kHz.
3. Retains recordings containing exactly 8,000 samples.
4. Encodes the word labels using `LabelEncoder`.
5. Converts labels into one-hot encoded vectors.
6. Reshapes the audio data into the CNN input format.

The resulting input shape is:

```python
(8000, 1)
```

**Feature representation:** The CNN uses raw audio waveforms as its input. Although MFCCs and spectrograms are discussed as audio-processing techniques, they are not the input representation used by the implemented CNN.

## CNN Architecture

The project implements a one-dimensional Convolutional Neural Network designed to learn patterns directly from audio waveforms.

The architecture contains five convolutional blocks:

| Layer | Filters | Kernel Size | Activation |
|---|---:|---:|---|
| Conv1D | 8 | 13 | ReLU |
| Conv1D | 16 | 11 | ReLU |
| Conv1D | 32 | 9 | ReLU |
| Conv1D | 64 | 7 | ReLU |
| Conv1D | 128 | 5 | ReLU |

Each convolutional block is followed by max pooling and dropout.

The convolutional layers are followed by:

- Flatten layer
- Dense layer with 256 units
- Dense layer with 128 units
- Dense layer with 64 units
- Softmax output layer for 30 classes

**Total trainable parameters:** 1,088,062

## Model Training

The model uses the following training configuration:

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Loss function | Categorical Crossentropy |
| Evaluation metric | Accuracy |
| Batch size | 100 |
| Maximum epochs | 100 |
| Early stopping monitor | Validation loss |
| Early stopping patience | 15 |
| Minimum improvement | 0.0001 |

Early stopping is used to stop training when validation loss no longer improves sufficiently.

The notebook reports an 80% training split, with the remaining data divided into validation and test sets. The final validation/test split is not stratified.

## Model Performance

The saved notebook outputs report the following test-set performance:

<escape> </escape>

| Metric | Test Result |
|---|---:|
| Accuracy | **86%** |
| Macro Precision | 0.87 |
| Macro Recall | 0.85 |
| Macro F1-score | 0.86 |
| Weighted Precision | 0.86 |
| Weighted Recall | 0.86 |
| Weighted F1-score | 0.86 |

The best recorded validation accuracy was **87.16% at epoch 96**.

These figures are taken from the saved notebook outputs and have not been independently reproduced in a fresh environment.

## Repository Structure

```text
Speech-Recognition-Using-CNN/
│
├── Cw1_w1843754_FardinAhmed.ipynb
├── images/
├── Requirements.txt
└── README.md
```

- `Cw1_w1843754_FardinAhmed.ipynb` — notebook containing the preprocessing, CNN implementation, training, and evaluation workflow.
- `images/` — project images and visual materials.
- `Requirements.txt` — existing project requirements file.
- `README.md` — project documentation.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/fardinAhmed285/Speech-Recognition-Using-CNN.git
```

### 2. Navigate to the project directory

```bash
cd Speech-Recognition-Using-CNN
```

### 3. Install the Python dependencies

A starting installation command is:

```bash
pip install tensorflow keras librosa numpy scipy scikit-learn matplotlib seaborn
```

Depending on your Python environment, compatible package versions may need to be selected.

### 4. Configure the dataset

Download the Google Speech Commands dataset and configure the audio directory in the notebook to point to your dataset location.

If using Google Colab, mount Google Drive and update the dataset path accordingly.

### 5. Run the notebook

Open:

`Cw1_w1843754_FardinAhmed.ipynb`

Run the cells in sequence to perform preprocessing, build the CNN, train the model, and evaluate its performance.

## Model Output

The notebook includes code for saving the trained model in HDF5 format (`kws.h5`) and TensorFlow SavedModel format.

The generated model files depend on running the training workflow; they are not listed as part of the repository structure above.

## Future Improvements

Potential extensions to this project include:

- Experimenting with MFCC and Mel-spectrogram input features.
- Comparing CNN performance with recurrent neural networks and other architectures.
- Applying data augmentation to improve robustness.
- Improving recognition accuracy through hyperparameter tuning.
- Testing the model on unseen real-world audio recordings.
- Developing a real-time speech recognition interface.

## Author

**Fardin Ahmed**

GitHub: [@fardinAhmed285](https://github.com/fardinAhmed285)

## Project Repository

[Speech Recognition Using CNN — GitHub](https://github.com/fardinAhmed285/Speech-Recognition-Using-CNN)

---

*This project was developed for learning and experimentation with deep learning techniques for audio classification and speech recognition.*