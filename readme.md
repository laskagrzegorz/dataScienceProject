# Data science project on genre classification

The aim of my project is to experiment with few classification models, 
find best choice hyperparameters for each of them and write a short report 
to audience with significant statistical background (or potentially CTO of music platform) 
about the performance of each of the model and my recommendation.

**Data sources**

The spotify data was sourced from HuggingFace. The link to the website is: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset.
The description of the data set and all the variables and its source is given on the website. 

## 1_data/

This directory contains all raw and derived dataset. Derived directory contains full dataset that was processes and encoded
(selected_spotify_data.csv), datasets X_train_normalised.csv, X_test_normalised.csv which are the subsets of features (predictors)
separated into train and test with corresponding y_train.csv and y_train.csv which contain the encoded labels. The encoding is as follows:

- Encoded label: 0, Original label: Electronic Dance Music
- Encoded label: 1, Original label: Funk and Disco
- Encoded label: 2, Original label: Hip-Hop and R&B
- Encoded label: 3, Original label: Latin & Reggae/Dancehall
- Encoded label: 4, Original label: Other
- Encoded label: 5, Original label: Pop
- Encoded label: 6, Original label: Rock