# Data science project on genre classification

Welcome to my data science project focused on genre classification using Spotify data.
This project aims to explore various classification models, fine-tune their hyperparameters, and present a comprehensive 
analysis of their performance. Our goal is to provide insights that could be valuable for stakeholders with significant statistical background with interested in music 
platforms, including potential CTOs.

**Data sources**

Our project leverages the Spotify tracks dataset available on HuggingFace, which provides rich metadata for thousands of songs across different genres. 
We have processed and encoded this data to facilitate machine learning model training and evaluation.  The link to the website is: 
https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset. I will use danceability, energy, loudness, acousticness, instrumentalness, liveness, valence and tempo as my predictor features.

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

This directory also contains `data_processing.py` file that contains all pre-processing.

### 2_analysis/

 Holds the Python scripts where we implement and evaluate various classification models and we do basic exploratory analysis.
 Each directory in `training_and_hyperparameter_tuning` directory contains the experimentation script, best model saved 
 with pickle and a plot representing the hyperparameter tuning.

### 3_outputs/

 Stores the output of our analyses, the final report and its overleaf directory to reproduce the report. Additionally it contains
python script showing for to properly import trained model and make a prediction based on new data introduced by user.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You will also need to install the required libraries listed in `requirements.txt`.

### Usage

To run the analysis, execute the main script located in the `2_analysis` directory. Follow the instructions within the script or accompanying documentation for detailed steps.

## Contributing

Contributions are welcome. Please feel free to submit pull requests or open issues if you encounter any problems or have suggestions for improvement.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or need further assistance, please contact us via email or through the GitHub issue tracker.

