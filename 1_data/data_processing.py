import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# =======================================================================================
# Data pre-processing for training later models. I select features of interest,
# divide genres into categories, standardise the features and encode music_category
# =======================================================================================

# -------------------------------------------------------------------------------
# Load and select features of interest
# -------------------------------------------------------------------------------

# Load in the dataset
spotify_data = pd.read_csv('raw/spotify_dataset.csv', index_col=0)

# Select columns of interest
selected_spotify_data = spotify_data[['danceability', 'energy', 'loudness',
                                      'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                                      'valence', 'tempo', 'track_genre']]

# Drop any Na's
selected_spotify_data = selected_spotify_data.dropna()

# Genre to category mapping for playlist creation
genre_to_category = {
    # EDM
    'edm': 'Electronic Dance Music',
    'house': 'Electronic Dance Music',
    'electro': 'Electronic Dance Music',
    'trance': 'Electronic Dance Music',
    'techno': 'Electronic Dance Music',
    'dubstep': 'Electronic Dance Music',
    'drum-and-bass': 'Electronic Dance Music',
    'deep-house': 'Electronic Dance Music',
    'detroit-techno': 'Electronic Dance Music',
    'minimal-techno': 'Electronic Dance Music',
    'progressive-house': 'Electronic Dance Music',
    'breakbeat': 'Electronic Dance Music',

    # Rock
    'alt-rock': 'Rock',
    'rock': 'Rock',
    'indie': 'Rock',
    'indie-pop': 'Rock',
    'punk': 'Rock',
    'punk-rock': 'Rock',
    'hard-rock': 'Rock',
    'metal': 'Rock',
    'heavy-metal': 'Rock',
    'black-metal': 'Rock',
    'death-metal': 'Rock',
    'grunge': 'Rock',

    # Hip-Hop and R&B
    'hip-hop': 'Hip-Hop and R&B',
    'r-n-b': 'Hip-Hop and R&B',
    'trap': 'Hip-Hop and R&B',

    # Pop
    'pop': 'Pop',
    'electro-pop': 'Pop',
    'synth-pop': 'Pop',
    'k-pop': 'Pop',
    'pop-film': 'Pop',
    'power-pop': 'Pop',

    # Latin & Reggae/Dancehall
    'latin': 'Latin & Reggae/Dancehall',
    'reggaeton': 'Latin & Reggae/Dancehall',
    'salsa': 'Latin & Reggae/Dancehall',
    'samba': 'Latin & Reggae/Dancehall',
    'reggae': 'Latin & Reggae/Dancehall',
    'dancehall': 'Latin & Reggae/Dancehall',

    # Funk and Disco
    'funk': 'Funk and Disco',
    'disco': 'Funk and Disco',
    'groove': 'Funk and Disco',
}

# Map each track to a category
selected_spotify_data['music_category'] = selected_spotify_data['track_genre'].apply(
    lambda x: genre_to_category.get(x, 'Other'))

# Filter out 20k samples of 'Other' category
other_category_samples = selected_spotify_data[selected_spotify_data['music_category'] == 'Other'].sample(n=20000, random_state=42)

# Concatenate the filtered 'Other' samples with the rest of the data
selected_spotify_data_filtered = pd.concat([selected_spotify_data[selected_spotify_data['music_category'] != 'Other'], other_category_samples])

# -------------------------------------------------------------------------------
# Standardise features, encode music_category and save all the files
# -------------------------------------------------------------------------------

# Save the selected data to a CSV file
selected_spotify_data.to_csv('../1_data/derived/selected_spotify_data.csv')

# Split data into features (X) and target variable (y)
X = selected_spotify_data.drop(columns=['music_category', 'track_genre']).to_numpy()
y = selected_spotify_data['music_category']

# Encode the categorical target variable y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create a StandardScaler instance
scaler = StandardScaler(with_mean=True)

# Split data into training testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=100)

# Fit and transform train data
X_train_normalised = scaler.fit_transform(X_train)

# Transform test data using scalar trained on train data
X_test_normalised = scaler.transform(X_test)

# Save X_train_normalised to CSV
pd.DataFrame(X_train_normalised).to_csv('../1_data/derived/X_train_normalised.csv', index=False)

# Save X_test_normalised to CSV
pd.DataFrame(X_test_normalised).to_csv('../1_data/derived/X_test_normalised.csv', index=False)

# Save y_train to CSV
pd.DataFrame(y_train).to_csv('../1_data/derived/y_train.csv', index=False, header=['track_genre_encoded'])

# Save y_test to CSV
pd.DataFrame(y_test).to_csv('../1_data/derived/y_test.csv', index=False, header=['track_genre_encoded'])

# Save scaler object using pickle
with open('../1_data/derived/scaler_values.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label_encoder object using pickle
with open('../1_data/derived/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# -------------------------------------------------------------------------------
# music_category Encoding
# -------------------------------------------------------------------------------

# Encoded label: 0, Original label: Electronic Dance Music
# Encoded label: 1, Original label: Funk and Disco
# Encoded label: 2, Original label: Hip-Hop and R&B
# Encoded label: 3, Original label: Latin & Reggae/Dancehall
# Encoded label: 4, Original label: Other
# Encoded label: 5, Original label: Pop
# Encoded label: 6, Original label: Rock

