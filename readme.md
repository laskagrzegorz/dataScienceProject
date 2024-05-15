# Data science project on genre classification

Welcome to my data science project focused on genre classification using Spotify data.
This project aims to explore various classification models, fine-tune their hyperparameters, and present a comprehensive 
analysis of their performance. Our goal is to provide insights that could be valuable for stakeholders interested in music 
platforms, including potential CTOs.

**Data sources**

Our project leverages the Spotify tracks dataset available on HuggingFace, which provides rich metadata for thousands of songs across different genres. 
We have processed and encoded this data to facilitate machine learning model training and evaluation.  The link to the website is: 
https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset.

## Repository Structure

### 1_data/

Contains the raw and processed dataset, including feature subsets (`X_train_normalised.csv`, `X_test_normalised.csv`) and encoded labels (`y_train.csv`, `y_test.csv`).
It also includes pickle files for data transformation (`scaler_values.pkl`) and label encoding (`label_encoder.pkl`). The encoding is as follows:

- Encoded label: 0, Original label: Electronic Dance Music
- Encoded label: 1, Original label: Funk and Disco
- Encoded label: 2, Original label: Hip-Hop and R&B
- Encoded label: 3, Original label: Latin & Reggae/Dancehall
- Encoded label: 4, Original label: Other
- Encoded label: 5, Original label: Pop
- Encoded label: 6, Original label: Rock

This directory also contains data_processing.py file that contains all pre-processing.
