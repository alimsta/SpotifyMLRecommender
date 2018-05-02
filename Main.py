import spotipy
import spotipy.util as util
from Helpers import get_training_data, classify_tester, recommend, plot


class SpotifyML:

    # Program will run on object instantiation
    def __init__(self):
        cid = ""         # Enter your own client_id from Spotify Web API
        secret = ""      # Enter your own secret from Spotify Web API
        username = ""    # Enter your own username

        scope = 'playlist-modify-private playlist-modify-public playlist-read-private user-library-read'
        # Must enter your specified redirect URI stored in Project Info on Spotify developer site.
        token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri="")

        if token:
            sp = spotipy.Spotify(auth=token)
            print("Token acquired.")
        else:
            print("Token not acquired.")

        # Enter your 'good' playlist id and 'bad' playlist id into get_training_data, respectively.
        trainingData = get_training_data("", "", username, sp)
        plot(trainingData)
        classify_tester(trainingData)
        # Enter the id of the playlist you want this program to recommend songs from.
        playlist = ""
        self.songs = recommend(trainingData, username, playlist, sp)

    # Call str(object) magic method to print recommended songs
    def __str__(self):
        temp = self.songs
        string = ""
        for song in temp:
            string = string + song + "\n"
        return string


test = SpotifyML()
print(str(test))