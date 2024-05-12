import pandas as pd

# Load in the dataset
spotify_data = pd.read_csv('../1_data/raw/dataset.csv', index_col=0)

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

# Save the selected data to a CSV file
selected_spotify_data.to_csv('../1_data/derived/selected_spotify_data.csv')

