import librosa

audio_path = 'song_examples/one_more_time.ogg'
y, sr = librosa.load(audio_path)

y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, units='time')

print(beats)
