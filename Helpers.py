import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Get songs and their ids for the given playlist
def get_ids(user, playlist, sp):
    playlist = sp.user_playlist(user, playlist)
    tracks = playlist["tracks"]
    songs = tracks["items"]
    while tracks['next']:
        tracks = sp.next(tracks)
        for item in tracks["items"]:
            songs.append(item)
    ids = []
    for i in range(len(songs)):
        ids.append(songs[i]['track']['id'])
    print(len(songs))
    return ids, songs

# Get audio features for each song
def get_features(ids, songs, sp, boolean=None):
    features = []
    j = 0
    for i in range(0, len(ids), 50):
        try:
            audio_features = sp.audio_features(ids[i:i + 50])
            for track in audio_features:
                print(track)
                features.append(track)
                j = j + 1
                if boolean:
                    features[-1]['target'] = 1
                else:
                    features[-1]['target'] = 0
        except AttributeError:
            continue
    return features


def get_training_data(good, bad, username, sp):
    bad_ids = get_ids(username, bad, sp)
    good_ids = get_ids(username, good, sp)

    x = get_features(bad_ids[0], bad_ids[1], sp, False)
    y = get_features(good_ids[0], good_ids[1], sp, True)

    features = x + y

    training_data = pd.DataFrame(features)
    return training_data

# Testing different classifiers for accuracy
def classify_tester(training_data):

    train, test = train_test_split(training_data, test_size=0.20, shuffle=True)
    print()
    print("Training size: {}, Test size: {}".format(len(train), len(test)))

    features = ["tempo", "danceability", "loudness", "valence", "acousticness", "key", "speechiness", "duration_ms"]
    x_train = train[features]
    y_train = train["target"]

    x_test = test[features]
    y_test = test["target"]

    dtc = DecisionTreeClassifier(min_samples_split=100)
    dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)
    score = accuracy_score(y_test, pred) * 100
    print("Accuracy for DecisionTreeClassifier: ", round(score, 1), "%")

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(x_train, y_train)
    pred = kmeans.predict(x_test)
    score = accuracy_score(y_test, pred) * 100
    print("Accuracy for KMeans: ", round(score, 1), "%")

    knn = KNeighborsClassifier(5)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    score = accuracy_score(y_test, pred) * 100
    print("Accuracy for K-Nearest Neighbors: ", round(score, 1), "%")

    ada = AdaBoostClassifier(n_estimators=150)
    ada.fit(x_train, y_train)
    pred = ada.predict(x_test)
    score = accuracy_score(y_test, pred) * 100
    print("Accuracy for AdaBoost: ", round(score, 1), "%")

    gbc = GradientBoostingClassifier(n_estimators=175, learning_rate=.15, max_depth=10, random_state=0)
    gbc.fit(x_train, y_train)
    pred = gbc.predict(x_test)
    score = accuracy_score(y_test, pred) * 100
    print("Accuracy for GradientBoost: ", round(score, 1), "%")

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    qda_pred = qda.predict(x_test)
    score = accuracy_score(y_test, qda_pred) * 100
    print("Accuracy for Discriminant Analysis: ", round(score, 1), "%")

# Decided to use GradientBoostingClassifier given accuracy results
def recommend(training_data, user, playlist, sp):
    train, test = train_test_split(training_data, test_size=0.20, shuffle=True)

    features = ["tempo", "danceability", "loudness", "valence", "acousticness", "key", "speechiness", "duration_ms"]
    x_train = train[features]
    y_train = train["target"]

    x_test = test[features]

    gbc = GradientBoostingClassifier(n_estimators=175, learning_rate=.15, max_depth=10, random_state=0)
    gbc.fit(x_train, y_train)
    gbc.predict(x_test)

    playlist = sp.user_playlist(user, playlist)

    new_tracks = playlist["tracks"]
    new_songs = new_tracks["items"]
    while new_tracks['next']:
        new_tracks = sp.next(new_tracks)
        for song in new_tracks["items"]:
            new_songs.append(song)

    new_song_ids = []
    print()
    print("Number of songs to check: " + str(len(new_songs)))
    for i in range(len(new_songs)):
        new_song_ids.append(new_songs[i]['track']['id'])

    new_features = []
    j = 0
    for i in range(0, len(new_song_ids), 50):
        try:
            audio_features = sp.audio_features(new_song_ids[i:i + 50])
            for track in audio_features:
                track['song_title'] = new_songs[j]['track']['name']
                track['artist'] = new_songs[j]['track']['artists'][0]['name']
                j = j + 1
                new_features.append(track)
        except AttributeError:
            continue

    data = pd.DataFrame(new_features)
    pred = gbc.predict(data[features])

    print("SpotifyML recommends these songs:")
    rec_songs = 0
    i = 0
    songs = []

    for song in results_generator(pred, i):
        songs.append("Song: " + data["song_title"][song] + " - " + data["artist"][song])
        rec_songs += 1
    print("Number of songs recommended: " + str(rec_songs))
    return songs


def results_generator(pred, i):
    for p in pred:
        if p == 1:
            yield i
        i = i + 1

# Plot to compare good and bad playlist data
def plot(training_data):
    # Plot test and training data
    fig2 = plt.figure(figsize=(10, 10))

    # Danceability
    pos_dance = training_data[training_data['target'] == 1]['danceability']
    neg_dance = training_data[training_data['target'] == 0]['danceability']
    ax3 = fig2.add_subplot(431)
    ax3.set_title('Danceability')
    ax3.set_ylabel('Count')
    pos_dance.hist(alpha=0.5, bins=30)
    fig2.add_subplot(431)
    neg_dance.hist(alpha=0.5, bins=30)

    # Duration_ms
    pos_duration = training_data[training_data['target'] == 1]['duration_ms']
    neg_duration = training_data[training_data['target'] == 0]['duration_ms']
    ax5 = fig2.add_subplot(432)
    ax5.set_title('Duration')
    ax5.set_ylabel('Count')
    pos_duration.hist(alpha=0.5, bins=30)
    fig2.add_subplot(432)
    neg_duration.hist(alpha=0.5, bins=30)

    # Loudness
    pos_loudness = training_data[training_data['target'] == 1]['loudness']
    neg_loudness = training_data[training_data['target'] == 0]['loudness']
    ax7 = fig2.add_subplot(433)
    ax7.set_title('Loudness')
    ax7.set_ylabel('Count')
    pos_loudness.hist(alpha=0.5, bins=30)
    fig2.add_subplot(433)
    neg_loudness.hist(alpha=0.5, bins=30)

    # Speechiness
    pos_speechiness = training_data[training_data['target'] == 1]['speechiness']
    neg_speechiness = training_data[training_data['target'] == 0]['speechiness']
    ax9 = fig2.add_subplot(434)
    ax9.set_title('Speechiness')
    ax9.set_ylabel('Count')
    pos_speechiness.hist(alpha=0.5, bins=30)
    fig2.add_subplot(434)
    neg_speechiness.hist(alpha=0.5, bins=30)

    # Valence
    pos_valence = training_data[training_data['target'] == 1]['valence']
    neg_valence = training_data[training_data['target'] == 0]['valence']
    ax11 = fig2.add_subplot(435)
    ax11.set_title('Valence')
    ax11.set_ylabel('Count')
    pos_valence.hist(alpha=0.5, bins=30)
    fig2.add_subplot(435)
    neg_valence.hist(alpha=0.5, bins=30)

    # Energy
    pos_energy = training_data[training_data['target'] == 1]['energy']
    neg_energy = training_data[training_data['target'] == 0]['energy']
    ax13 = fig2.add_subplot(436)
    ax13.set_title('Energy')
    ax13.set_ylabel('Count')
    pos_energy.hist(alpha=0.5, bins=30)
    fig2.add_subplot(436)
    neg_energy.hist(alpha=0.5, bins=30)

    # Key
    pos_key = training_data[training_data['target'] == 1]['key']
    neg_key = training_data[training_data['target'] == 0]['key']
    ax15 = fig2.add_subplot(437)
    ax15.set_title('Key')
    ax15.set_ylabel('Count')
    pos_key.hist(alpha=0.5, bins=30)
    fig2.add_subplot(437)
    neg_key.hist(alpha=0.5, bins=30)

    # Tempo
    pos_tempo = training_data[training_data['target'] == 1]['tempo']
    neg_tempo = training_data[training_data['target'] == 0]['tempo']
    ax17 = fig2.add_subplot(438)
    ax17.set_title('Tempo')
    ax17.set_ylabel('Count')
    pos_tempo.hist(alpha=0.5, bins=30)
    fig2.add_subplot(438)
    neg_tempo.hist(alpha=0.5, bins=30)

    # Acousticness
    pos_acousticness = training_data[training_data['target'] == 1]['acousticness']
    neg_acousticness = training_data[training_data['target'] == 0]['acousticness']
    ax19 = fig2.add_subplot(439)
    ax19.set_title('Acoustic')
    ax19.set_ylabel('Count')
    pos_acousticness.hist(alpha=0.5, bins=30)
    fig2.add_subplot(439)
    neg_acousticness.hist(alpha=0.5, bins=30)

    # Instrumentalness
    pos_instrumentalness = training_data[training_data['target'] == 1]['instrumentalness']
    neg_instrumentalness = training_data[training_data['target'] == 0]['instrumentalness']
    ax21 = fig2.add_subplot(4, 3, 10)
    ax21.set_title('Instrumentalness')
    ax21.set_ylabel('Count')
    pos_instrumentalness.hist(alpha=0.5, bins=30)
    fig2.add_subplot(438)
    neg_instrumentalness.hist(alpha=0.5, bins=30)

    plt.show()