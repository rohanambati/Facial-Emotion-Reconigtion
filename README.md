# Facial Emotion Recognition

A convolutional neural network (CNN) model to classify facial expressions into seven emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using the FER2013 dataset.

## Technologies & Libraries

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- scikit-learn
- Pillow

## Dataset

The FER2013 dataset is provided as a CSV where each row contains:

- `emotion`: integer label 0–6  
- `pixels`: space-separated list of 48×48 = 2304 grayscale pixel values  
- `Usage`: one of `Training`, `PublicTest`, `PrivateTest`

Source: [Kaggle – nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition/data)

## Methodology & Workflow

### 1. Load & Inspect Data
- Read the CSV into a DataFrame.  
- Examine shapes and class distributions.

### 2. Preprocessing
- Split into training, validation (`PublicTest`), and test (`PrivateTest`) sets.  
- Convert `pixels` strings to NumPy arrays, reshape to `(48, 48, 1)`.  
- Normalize pixel values to `[0, 1]`.  
- One-hot encode the emotion labels.

### 3. Model Architecture
```plaintext
Conv2D → BatchNorm → ReLU → MaxPool
Conv2D → BatchNorm → ReLU → MaxPool
Conv2D → BatchNorm → ReLU → MaxPool
Flatten
Dense(128) → BatchNorm → ReLU → Dropout(0.6)
Dense(7) → Softmax
```
- **Loss:** categorical_crossentropy  
- **Optimizer:** Adam  
- **Metrics:** accuracy

### 4. Training
- Train for up to N epochs (as defined in notebook).  
- Use `ModelCheckpoint` to save the best weights.

### 5. Evaluation & Visualization
- Plot training/validation loss & accuracy vs. epochs.  
- Display confusion matrix.  
- (Optional) ROC & Precision–Recall curves for multiclass.

### 6. Inference
- Load saved model.  
- Preprocess an input image.  
- Predict the emotion class.

## Usage
1. Clone the repo.  
2. Place `fer2013.csv` (or `emotional_dataset.csv`) in the project root.  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open and run `Facial Emotion Recognition.ipynb` in Jupyter or Colab.

## Results

![Results Screenshot](https://i.ibb.co/ghN34dC/image.png)

*(Include final accuracy, sample predictions, and plots here.)*

## License
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
