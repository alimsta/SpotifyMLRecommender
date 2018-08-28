# SpotifyMLRecommender

This program uses Spotipy and Scikit-learn to recommend songs from a given playlist by applying a GradientBoostingClassifier on song audio features trained on two playlists: one of ‘good’ songs and one of ‘bad’ songs. The program also uses pandas dataframes to store audio data and matplotlib to visualize how the training and test data compare. The SpotifyML class, which is instantiated to run the entire program, has a str() magic method to return the recommended songs as a string.

### Required Libraries

scikit-learn, matplotlib, pandas, spotipy
