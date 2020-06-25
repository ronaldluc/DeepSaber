from pathlib import Path

import gensim
from tensorflow.keras.models import Model

from predict.compute import zip_folder, update_generated_metadata, save_generated_beatmaps, \
    copy_folder_contents, create_beatmap_dfs
from utils.functions import create_word_mapping
from utils.types import Config


def generate_complete_beatmaps(beatmap_folder: Path, output_folder: Path, stateful_model: Model, config: Config):
    gen_folder = output_folder / f'{beatmap_folder.name}_generated'
    gen_folder.mkdir(parents=True, exist_ok=True)

    action_model = gensim.models.KeyedVectors.load(str(config.dataset.action_word_model_path))
    word_id_dict = create_word_mapping(action_model)

    beatmap_dfs = create_beatmap_dfs(stateful_model, action_model, word_id_dict, beatmap_folder, config)

    copy_folder_contents(beatmap_folder, gen_folder)
    save_generated_beatmaps(gen_folder, beatmap_dfs, action_model, word_id_dict, config)

    update_generated_metadata(gen_folder, beatmap_folder, config)

    zip_folder(gen_folder)

# if __name__ == '__main__':
#     gen_new_beat_map_path = '../data/new_dataformat/4ede/'
#     config = Config()
#     #
#     # df1 = songs2dataset([gen_new_beat_map_path, ], config)
#     #
#     # df2 = process_song_folder(gen_new_beat_map_path, config)
#     # config.beat_preprocessing.snippet_window_length = len(df2)
#     #
#     # seq = BeatmapSequence(df2, config)
#     # # ['name', 'difficulty', 'snippet', 'time']
#     # print('done')
#
#     path = '../data/temp/beatmap_df.pkl'
#     df = pd.read_pickle(path)
#     df2beatmap(df, config)
#     print(df)
