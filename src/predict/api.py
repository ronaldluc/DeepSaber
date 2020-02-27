from tensorflow.keras.models import Model

from process.api import songs2dataset
from process.compute import process_song_folder
from train.model import create_model
from train.sequence import BeatmapSequence
from utils.types import Config


def create_beat_map(model: Model, path, config: Config):
    df = process_song_folder(path, config)
    # df.

    config.beat_preprocessing['snippet_window_length'] = len(df)
    config.training['batch_size'] = 1
    seq = BeatmapSequence(df, config)

    stateful_model = create_model(seq, True, config)

    stateful_model.set_weights(model.get_weights())

    # Straight prediction
    # res = model.predict(seq)

    for i in range(len(df)):
        cur = {name: val[:, i:i+1] for name, val in seq.data.items()}
        next = stateful_model.predict(cur)
        print(stateful_model.summary())
        pred = {name: val for name, val in zip(stateful_model.output_names, next)}
    #     Store predictions
    #     update cur with  prev_*

    # TODO: Processing as in BeatMapSequence without copying code?
    # use heat
    # data2JSON
    # return JSON
    return res


if __name__ == '__main__':
    gen_new_beat_map_path = '../data/new_dataformat/4ede/'
    config = Config()

    df1 = songs2dataset([gen_new_beat_map_path, ], config)

    df2 = process_song_folder(gen_new_beat_map_path, config)
    config.beat_preprocessing['snippet_window_length'] = len(df2)

    seq = BeatmapSequence(df2, config)
    # ['name', 'difficulty', 'snippet', 'time']
    print('done')
