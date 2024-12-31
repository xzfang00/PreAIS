### Instructions for Using Prediction R-Predictor

#### 1. Introduction

We have meticulously crafted a predictive tool named "Prediction R-Predictor" to vividly showcase the experimental outcomes of our research. This tool is anchored in Convolutional Neural Networks (CNN) and Deep Neural Networks (DNN) models, specifically designed for analyzing biological sequence data presented in FASTA format and identifying AI editing sites within these sequences. The tool encompasses two core models: Model 1, which focuses on analyzing RNA sequences of length 101, and Model 2, tailored for RNA sequences of length 51. Model 1 predicts outcomes by extracting k-mer features, while Model 2 leverages BPB features for its predictions. The interface layout and navigation have been thoughtfully designed to enhance user experience, ensuring a straightforward and intuitive operation process, which facilitates ease of data prediction. "Prediction R-Predictor" is intended to provide users with a practical and efficient tool to support experimental observations and deepen the interpretation of experimental data.

#### 2. Environment Setup

It is essential to ensure the installation of the following dependencies:

- Python 3.9
- Tkinter
- PyTorch
- Biopython
- NumPy

#### 3. Usage Steps

#### **model1：**

1. **Launching the Program**: The program can be initiated by running `predictor.py`, which will open the main window.

   ![1](https://github.com/xzfang00/PreAIS/blob/main/img/1.png)

2. **Uploading a File**: The “Upload File” button should be clicked to select a FASTA file containing biological sequences.

   ![2](https://github.com/xzfang00/PreAIS/blob/main/img/2.png)

3. **Conducting Predictions**: The “Predict” button should be clicked, after which the program will automatically read the file, perform k-mer feature extraction, and subsequently utilize the pretrained model for prediction.

4. **Viewing Results**: Upon completion of the predictions, a results window will be displayed, showcasing the sequences, prediction outcomes (positive/negative), and corresponding probabilities.

   ![3](https://github.com/xzfang00/PreAIS/blob/main/img/3.png)
   
   #### **model2：**
   
   1. **Launching the Program**: The program can be initiated by running `predictor.py`, which will open the main window.
   
      ![1](https://github.com/xzfang00/PreAIS/blob/main/img/1.png)
   
   2. **Uploading a File**: The “Upload File” button should be clicked to select a FASTA file containing biological sequences.
   
      ![2](https://github.com/xzfang00/PreAIS/blob/main/img/2.png)
   
   3. **Conducting Predictions**: The “Predict” button should be clicked, after which the program will automatically read the file, perform BPB feature extraction, and subsequently utilize the pretrained model for prediction.
   
   4. **Viewing Results**: Upon completion of the predictions, a results window will be displayed, showcasing the sequences, prediction outcomes (positive/negative), and corresponding probabilities.
   
      ![3](https://github.com/xzfang00/PreAIS/blob/main/img/4.png)

#### 4. Important Notes

- It is imperative to ensure that the uploaded file is in the correct format (FASTA).
- Caution is advised when interpreting prediction results with probability values close to 0.5.

For further inquiries, please refer to the documentation or contact technical support at xzfang00@126.com.
