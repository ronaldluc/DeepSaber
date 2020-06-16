from pathlib import Path

import gensim
import logging
import pandas as pd

storage_folder = Path('../../data/new_datasets')


def eval_model(**kwargs):
    kwargs = {key: int(val) for key, val in kwargs.items()}
    kwargs['size'] = 2 ** kwargs['size']
    model = gensim.models.FastText(corpus_file=str(storage_folder / 'train_text.cor'), **kwargs, workers=12)

    res = model.wv.evaluate_word_analogies(storage_folder / 'beat_analogies.txt')

    # correct = sum([len(section['correct']) for section in res[1]])
    # incorrect = sum([len(section['incorrect']) for section in res[1]])
    # total_sum = correct + incorrect
    # print(f'Correct {correct:7} | incorrect {incorrect:7} | total {total_sum:7} | acc {correct/total_sum:7}')
    return res[0]


def main():
    from bayes_opt import BayesianOptimization
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    # Bounded region of parameter space

    bool_ = (0.1, 1.9)
    pbounds = {
        'size': (4, 9),  # log int
        'window': (1, 7),  # int
        'iter': (1.1, 50),  # int  : Number of iterations (epochs) over the corpus.
        'sg': bool_,  # bool : skip-gram if `sg=1`, otherwise CBOW.
        'hs': bool_,  # bool : If 1, hierarchical softmax will be used for model training.
        #        If set to 0, and `negative` is non-zero, negative sampling will be used.
        # 'sample': (0, 1e-5),   # float: The threshold for configuring which higher-frequency words are randomly downsampled,
        # 'negative': (0, 20),   # int  : If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        'cbow_mean': bool_,
        # bool : If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        'min_n': (2, 5),  # int  : Minimum length of char n-grams to be used for training word representations.
        'max_n': (3, 9),
        # int  : Max length of char ngrams to be used for training word representations. Set `max_n` to be lesser than `min_n` to avoid char ngrams being used.
        'word_ngrams': bool_,  # bool : If 1, uses enriches word vectors with subword(n-grams) information.
        # If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.
    }

    optimizer = BayesianOptimization(
        f=eval_model,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=80,
        n_iter=500,
    )

    rdf = pd.DataFrame(optimizer.res)
    fasttext_rdf = rdf.loc[rdf.params.apply(lambda x: x['word_ngrams'] >= 1)]
    word2vec_rdf = rdf.loc[rdf.params.apply(lambda x: x['word_ngrams'] < 1)]
    fasttext_params = rdf.loc[fasttext_rdf['target'].idxmax()].params
    word2vec_params = rdf.loc[word2vec_rdf['target'].idxmax()].params
    params = rdf.loc[rdf['target'].idxmax()].params

    print(params)
    rdf.to_csv('models_hyperparams.csv')


if __name__ == '__main__':
    main()
