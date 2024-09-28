### Instructions for Using Prediction R-Predictor

#### 1. Introduction

To provide an interactive experience for the intuitive demonstration of the experimental results of this study, a predictive tool has been developed for the analysis of input RNA data. Prediction R-Predictor is a tool based on CNN/DNN models, designed for the identification and prediction of AI editing sites in biological sequences formatted in FASTA. This tool extracts k-mer features and outputs predictive results. The interface layout and navigation design have been carefully considered to enhance user experience, ensuring that operations are straightforward and intuitive, thereby facilitating ease of use for data predictions. This predictive tool aims to provide users with a practical instrument to support experimental observations and promote the interpretation of experimental data.

#### 2. Environment Setup

It is essential to ensure the installation of the following dependencies:

- Python 3.x
- Tkinter
- PyTorch
- Biopython
- NumPy

#### 3. Usage Steps

1. **Launching the Program**: The program can be initiated by running `predictor.py`, which will open the main window.

   ![1](img\1.png)

2. **Uploading a File**: The “Upload File” button should be clicked to select a FASTA file containing biological sequences.

   ![2](.\img\2.png)

3. **Conducting Predictions**: The “Predict” button should be clicked, after which the program will automatically read the file, perform k-mer feature extraction, and subsequently utilize the pretrained model for prediction.

4. **Viewing Results**: Upon completion of the predictions, a results window will be displayed, showcasing the sequences, prediction outcomes (positive/negative), and corresponding probabilities.

   ![3](.\img\3.png)

#### 4. Important Notes

- It is imperative to ensure that the uploaded file is in the correct format (FASTA).
- Caution is advised when interpreting prediction results with probability values close to 0.5.

For further inquiries, please refer to the documentation or contact technical support at xzfang00@126.com.
