Leverages an ML classifier to predict songs that a user will like based on that user's Spotify playlists.
# SpotifyMLRecommender

This program uses Spotipy and Scikit-learn to recommend songs from a given playlist by using an ML Classifier on song audio features trained on two playlists: one of ‘good’ songs and one of ‘bad’ songs. The program also uses pandas and matplotlib to visualize how the audio features of the 'good' and 'bad' songs compare. The SpotifyML class, which is instantiated to run the entire program, also has a __str__ method, which returns the recommended songs as a string.

### Required Libraries

Scikit-learn, matplotlib, pandas, spotipy
