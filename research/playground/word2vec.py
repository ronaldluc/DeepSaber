r"""
Word2Vec Model
==============

Introduces Gensim's Word2Vec model and demonstrates its use on the Lee Corpus.

"""

import logging
from itertools import product
from pathlib import Path
from typing import List, Tuple, Any

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

global i

# wv = api.load('word2vec-google-news-300')
#
# for i, word in enumerate(wv.vocab):
#     if i == 10:
#         break
#     print(word)
#
# vec_king = wv['king']
#
# try:
#     vec_cameroon = wv['cameroon']
# except KeyError:
#     print("The word 'cameroon' does not appear in this model")
#
# pairs = [
#     ('car', 'minivan'),   # a minivan is a kind of car
#     ('car', 'bicycle'),   # still a wheeled vehicle
#     ('car', 'airplane'),  # ok, no wheels, but still a vehicle
#     ('car', 'cereal'),    # ... and so on
#     ('car', 'communism'),
# ]
# for w1, w2 in pairs:
#     print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))
#
# print(wv.most_similar(positive=['car', 'minivan'], topn=5))
#
# print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))

###############################################################################
# Training Your Own Model
# -----------------------


from gensim.test.utils import datapath
from gensim import utils


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


import gensim.models
import pandas as pd


def train_batch(corpus_path, models_folder, batch: pd.DataFrame):
    models = {
        'word2vec': gensim.models.Word2Vec,
        'fasttext': gensim.models.FastText
    }

    for i, settings in batch.iterrows():
        try:
            print(f'Training settings: {settings}')
            if settings['data'] == 'raw':
                model = models[settings['model']](corpus_file=str(corpus_path),
                                                  size=settings['size'],
                                                  workers=16, min_count=15)
            else:
                model = models[settings['model']](sentences=MyCorpus(corpus_path),
                                                  size=settings['size'],
                                                  workers=16, min_count=15)

            model_name = models_folder / f'{settings["model"]}-{settings["size"]}-{settings["data"]}.model'
            model.save(str(model_name))

            print('=' * 140)
            print(f'Saved {model_name}')
            print('=' * 140)
        except Exception as e:
            print(f'ERROR: {e} ' * 42)


def get_trained_model(corpus_path):
    print(f'Word2Vec implementation: {gensim.models.word2vec.FAST_VERSION} ')

    model = gensim.models.word2vec.Word2Vec(corpus_file=str(corpus_path), size=32, workers=32,
                                            max_vocab_size=4e5, min_count=5, iter=10,
                                            window=3)
    # model = gensim.models.FastText(corpus_file=str(corpus_path), size=128, workers=16,
    #                                max_vocab_size=4e5, min_count=5, iter=1,
    #                                window=2)

    return model


###############################################################################
# Storing and loading models
# --------------------------
#
# You'll notice that training non-trivial models can take time.  Once you've
# trained your model and it works as expected, you can save it to disk.  That
# way, you don't have to spend time training it all over again later.
#
# You can store/load models using the standard gensim methods:
#
import tempfile


def save_model(model):
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
        temporary_filepath = tmp.name
        model.save(temporary_filepath)


def load_model():
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
        temporary_filepath = tmp.name

        return gensim.models.Word2Vec.load(temporary_filepath)


###############################################################################
# which uses pickle internally, optionally ``mmap``\ ‘ing the model’s internal
# large NumPy matrices into virtual memory directly from disk files, for
# inter-process memory sharing.
#
# In addition, you can load models created by the original C tool, both using
# its text and binary formats::
#
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)
#   # using gzipped/bz2 input works too, no need to unzip
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
#

###############################################################################
# In the December 2016 release of Gensim we added a better way to evaluate semantic similarity.
#
# By default it uses an academic dataset WS-353 but one can create a dataset
# specific to your business based on it. It contains word pairs together with
# human-assigned similarity judgments. It measures the relatedness or
# co-occurrence of two words. For example, 'coast' and 'shore' are very similar
# as they appear in the same context. At the same time 'clothes' and 'closet'
# are less similar because they are related but not interchangeable.
#


###############################################################################
# .. Important::
#   Good performance on Google's or WS-353 test set doesn’t mean word2vec will
#   work well in your application, or vice versa. It’s always best to evaluate
#   directly on your intended task. For an example of how to use word2vec in a
#   classifier pipeline, see this `tutorial
#   <https://github.com/RaRe-Technologies/movie-plots-by-genre>`_.
#
def resume_training(model, corpus_path=None):
    sentences = MyCorpus(corpus_path)
    ###############################################################################
    # Online training / Resuming training
    # -----------------------------------
    #
    # Advanced users can load a model and continue training it with more sentences
    # and `new vocabulary words <online_w2v_tutorial.ipynb>`_:
    #
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    # cleaning up temporary file
    ###############################################################################
    # You may need to tweak the ``total_words`` parameter to ``wordvec()``,
    # depending on what learning rate decay you want to simulate.
    #
    # Note that it’s not possible to resume training with models generated by the C
    # tool, ``KeyedVectors.load_word2vec_format()``. You can still use them for
    # querying/similarity, but information vital for training (the vocab tree) is
    # missing there.

    return model


def compute_train_loss():
    ###############################################################################
    # Training Loss Computation
    # -------------------------
    #
    # The parameter ``compute_loss`` can be used to toggle computation of loss
    # while training the Word2Vec model. The computed loss is stored in the model
    # attribute ``running_training_loss`` and can be retrieved using the function
    # ``get_latest_training_loss`` as follows :
    #
    # instantiating and training the Word2Vec model
    model_with_loss = gensim.models.Word2Vec(
        sentences,
        min_count=1,
        compute_loss=True,
        hs=0,
        sg=1,
        seed=42
    )
    # getting the training loss value
    training_loss = model_with_loss.get_latest_training_loss()
    print(training_loss)


###############################################################################
# Benchmarks
# ----------
#
# Let's run some benchmarks to see effect of the training loss computation code
# on training time.
#
# We'll use the following data for the benchmarks:
#
# #. Lee Background corpus: included in gensim's test data
# #. Text8 corpus.  To demonstrate the effect of corpus size, we'll look at the
#    first 1MB, 10MB, 50MB of the corpus, as well as the entire thing.
#

import io

import gensim.models.word2vec
import gensim.downloader as api
import smart_open


def head(path, size):
    with smart_open.open(path) as fin:
        return io.StringIO(fin.read(size))


def generate_input_data():
    lee_path = datapath('lee_background.cor')
    ls = gensim.models.word2vec.LineSentence(lee_path)
    ls.name = '25kB'
    yield ls

    text8_path = api.load('text8').fn
    labels = ('1MB', '10MB', '50MB', '100MB')
    sizes = (1024 ** 2, 10 * 1024 ** 2, 50 * 1024 ** 2, 100 * 1024 ** 2)
    for l, s in zip(labels, sizes):
        ls = gensim.models.word2vec.LineSentence(head(text8_path, s))
        ls.name = l
        yield ls


def compare():
    global i
    ###############################################################################
    # We now compare the training time taken for different combinations of input
    # data and model training parameters like ``hs`` and ``sg``.
    #
    # For each combination, we repeat the test several times to obtain the mean and
    # standard deviation of the test duration.
    #
    # Temporarily reduce logging verbosity
    logging.root.level = logging.ERROR
    import time
    import numpy as np
    import pandas as pd
    train_time_values = []
    seed_val = 42
    sg_values = [0, 1]
    hs_values = [0, 1]
    fast = True
    if fast:
        input_data_subset = input_data[:3]
    else:
        input_data_subset = input_data
    for data in input_data_subset:
        for sg_val in sg_values:
            for hs_val in hs_values:
                for loss_flag in [True, False]:
                    time_taken_list = []
                    for i in range(3):
                        start_time = time.time()
                        w2v_model = gensim.models.Word2Vec(
                            data,
                            compute_loss=loss_flag,
                            sg=sg_val,
                            hs=hs_val,
                            seed=seed_val,
                        )
                        time_taken_list.append(time.time() - start_time)

                    time_taken_list = np.array(time_taken_list)
                    time_mean = np.mean(time_taken_list)
                    time_std = np.std(time_taken_list)

                    model_result = {
                        'train_data': data.name,
                        'compute_loss': loss_flag,
                        'sg': sg_val,
                        'hs': hs_val,
                        'train_time_mean': time_mean,
                        'train_time_std': time_std,
                    }
                    print("Word2vec model #%i: %s" % (len(train_time_values), model_result))
                    train_time_values.append(model_result)
    train_times_table = pd.DataFrame(train_time_values)
    train_times_table = train_times_table.sort_values(
        by=['train_data', 'sg', 'hs', 'compute_loss'],
        ascending=[False, False, True, False],
    )
    print(train_times_table)


def model2dict():
    global i, word
    ###############################################################################
    # Adding Word2Vec "model to dict" method to production pipeline
    # -------------------------------------------------------------
    #
    # Suppose, we still want more performance improvement in production.
    #
    # One good way is to cache all the similar words in a dictionary.
    #
    # So that next time when we get the similar query word, we'll search it first in the dict.
    #
    # And if it's a hit then we will show the result directly from the dictionary.
    #
    # otherwise we will query the word and then cache it so that it doesn't miss next time.
    #
    # re-enable logging
    logging.root.level = logging.INFO
    most_similars_precalc = {word: model.wv.most_similar(word) for word in model.wv.index2word}
    for i, (key, value) in enumerate(most_similars_precalc.items()):
        if i == 3:
            break
        print(key, value)
    ###############################################################################
    # Comparison with and without caching
    # -----------------------------------
    #
    # for time being lets take 4 words randomly
    #
    import time
    words = ['voted', 'few', 'their', 'around']
    ###############################################################################
    # Without caching
    #
    start = time.time()
    for word in words:
        result = model.wv.most_similar(word)
        print(result)
    end = time.time()
    print(end - start)
    ###############################################################################
    # Now with caching
    #
    start = time.time()
    for word in words:
        if 'voted' in most_similars_precalc:
            result = most_similars_precalc[word]
            print(result)
        else:
            result = model.wv.most_similar(word)
            most_similars_precalc[word] = result
            print(result)
    end = time.time()
    print(end - start)


###############################################################################
# Clearly you can see the improvement but this difference will be even larger
# when we take more words in the consideration.
#

###############################################################################
#
# Visualising the Word Embeddings
# -------------------------------
#
# The word embeddings made by the model can be visualised by reducing
# dimensionality of the words to 2 dimensions using tSNE.
#
# Visualisations can be used to notice semantic and syntactic trends in the data.
#
# Example:
#
# * Semantic: words like cat, dog, cow, etc. have a tendency to lie close by
# * Syntactic: words like run, running or cut, cutting lie close together.
#
# Vector relations like vKing - vMan = vQueen - vWoman can also be noticed.
#
# .. Important::
#   The model used for the visualisation is trained on a small corpus. Thus
#   some of the relations might not be so clear.
#

from sklearn.decomposition import IncrementalPCA  # inital reduction
from sklearn.decomposition import PCA  # inital reduction
from sklearn.manifold import TSNE  # final reduction
import numpy as np  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    # for word in model.wv.vocab:
    for word in model.wv.index2entity[:5000]:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # ipca = IncrementalPCA(n_components=10, batch_size=100)
    # vectors = ipca.fit_transform(vectors)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    # tsne = TSNE(n_components=num_dimensions, random_state=0)
    tsne = PCA(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(24, 24))
    anotate = min(450, len(labels))
    plt.scatter(x_vals[:anotate], y_vals[:anotate], alpha=0)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    # selected_indices = random.sample(indices, 25)
    for i in range(anotate):
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()


# noinspection PyBroadException
try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly


def visualize_model(model):
    x_vals, y_vals, labels = reduce_dimensions(model)
    plot_function(x_vals, y_vals, labels)


def project_words(model, words: List):
    num_dimensions = 2

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    # for word in model.wv.vocab:

    for i in range(0, len(words), 2):
        if words[i] in model.wv and words[i + 1] in model.wv:
            for word in words[i:i + 2]:
                vectors.append(model.wv[word])
                labels.append(word)
        else:
            print(f'Not found {words[i: i + 2]}')

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    tsne = PCA(n_components=num_dimensions, random_state=0, )
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]

    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(24, 24))
    anotate = min(450, len(labels))
    plt.scatter(x_vals[:anotate], y_vals[:anotate], alpha=0)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    # selected_indices = random.sample(indices, 25)
    for i in range(anotate):
        # plt.annotate(labels[i], (x_vals[i], y_vals[i]),             )
        plt.text(x_vals[i], y_vals[i], labels[i],
                 fontsize=26,
                 ha="center", va="center")

    for i in range(0, anotate, 2):
        offset = 0.05
        start = vectors[i] + offset * (vectors[i + 1] - vectors[i])
        end = (1 - offset * 2) * (vectors[i + 1] - vectors[i])
        plt.arrow(*start,
                  *end,
                  width=0.1,
                  alpha=0.5,
                  length_includes_head=True)

    plt.show()


###############################################################################
# Conclusion
# ----------
#
# In this tutorial we learned how to wordvec word2vec models on your custom data
# and also how to evaluate it. Hope that you too will find this popular tool
# useful in your Machine Learning tasks!
#
# Links
# -----
#
# - API docs: :py:mod:`gensim.models.word2vec`
# - `Original C toolkit and word2vec papers by Google <https://code.google.com/archive/p/word2vec/>`_.
#

def word_tuple2string(tuple: Tuple[Any]):
    if tuple[0] in 'LR' and 0 <= tuple[1] < 3 and 0 <= tuple[2] < 4 and 0 <= tuple[3] < 9:
        return ''.join([str(x) for x in tuple])
    return None


def create_analogies(path: Path):
    lines = []

    lines.append(': single-hand-translation')
    for translation in product(range(-2, 3), range(-3, 4)):
        if translation == (0, 0):  # remove identity translation
            continue
        lines.append(f': single-hand-translation {translation}')
        translation = '', *translation, 0
        # fn = lambda word_from, translation: [a + b for a, b in zip(word_from, translation)]
        for hand, rotation in product('LR', range(9)):
            for example_from in product(range(3), range(4)):
                example_from = hand, *example_from, rotation
                example_to = [a + b for a, b in zip(example_from, translation)]

                for question_from in product(range(3), range(4)):
                    question_from = hand, *question_from, rotation
                    question_to = [a + b for a, b in zip(question_from, translation)]
                    if example_from == question_from:   # skip identical
                        continue

                    analogy = [word_tuple2string(x) for x in [example_from, example_to, question_from, question_to]]
                    if None in analogy:
                        continue
                    lines.append(' '.join(analogy))

    lines.append(': single-hand-rotation')
    for rotation_change in range(-8, 9):
        if rotation_change == 0:  # remove identity rotation change
            continue
        translation = '', 0, 0, rotation_change
        for hand, rotation in product('LR', range(9)):
            for example_from in product(range(3), range(4)):
                example_from = hand, *example_from, rotation
                example_to = [a + b for a, b in zip(example_from, translation)]

                for question_from in product(range(3), range(4)):
                    question_from = hand, *question_from, rotation
                    question_to = [a + b for a, b in zip(question_from, translation)]
                    if example_from == question_from:   # skip identical
                        continue

                    analogy = [word_tuple2string(x) for x in [example_from, example_to, question_from, question_to]]
                    if None in analogy:
                        continue
                    lines.append(' '.join(analogy))

    lines.append(': single-hand-hand-swap')
    for hand_id in range(2):
        hand_from = 'LR'[hand_id]
        hand_to = 'RL'[hand_id]

        for example_from in product(range(3), range(4), range(9)):
            example_to = hand_to, *example_from
            example_from = hand_from, *example_from
            for question_from in product(range(3), range(4), range(9)):
                question_to = hand_to, *question_from
                question_from = hand_from, *question_from
                if example_from == question_from:   # skip identical
                    continue

                analogy = [word_tuple2string(x) for x in [example_from, example_to, question_from, question_to]]
                if None in analogy:
                    continue
                lines.append(' '.join(analogy))

    with open(path, 'w') as wf:
        wf.write('\n'.join(lines) + '\n')

    pass


def main():
    models_folder = Path('../../data/new_datasets/models')
    models_folder.mkdir(parents=True, exist_ok=True)
    corpus_path = Path('../../data/new_datasets/train_text.cor')

    model_path = str(models_folder / 'word2vec.model')

    # model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
    # model = resume_training(model, corpus_path)
    model = get_trained_model(corpus_path)
    # model.save(model_path, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2)
    # model = gensim.models.Word2Vec.load(model_path)

    print('Find similar:')
    find_similar = 'L000_L100_R010_R110 L000 L001 L100 R000 R001 R100'
    for word in find_similar.split():
        try:
            print(f'\t{word}')
            print(f'\t\t{model.wv.most_similar(positive=word, topn=5)}')
        except KeyError as e:
            print(e)

    # noinspection PyRedeclaration
    words = 'L000 L001 ' \
            'L100 L101 ' \
            'L010 L011 ' \
            'R000 R001 ' \
            'R100 R101 ' \
            'R010 R011 '.split()
    # words = 'program programmer ' \
    #         'architecture architect ' \
    #         'engine engineer'.split()
    # words = 'good bad easy hard high low windows linux apple IBM VMware google aws microsoft'.split()
    # project_words(model, words)
    # visualize_model(model)

    create_analogies(models_folder / 'beat_analogies.txt')

    # model.wv.evaluate_word_analogies()
    model.wv.evaluate_word_analogies(models_folder / 'beat_analogies.txt', restrict_vocab=5000)  # TODO: Try higher values
    # model.evaluate_word_pairs(datapath('wordsim353.tsv'))


if __name__ == '__main__':
    main()
