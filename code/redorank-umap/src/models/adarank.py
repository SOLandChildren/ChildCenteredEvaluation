"""
AdaRank algorithm adapted to use the implementation of NCSDCGScorer from https://arxiv.org/pdf/2308.15265

Base algorithm implementation adapted from https://github.com/rueycheng/AdaRank/blob/master/adarank.py
"""
import math
import numpy as np
import sklearn
import sys

from sklearn.utils import check_X_y
from src.metrics import NCSDCGScorerQid
from pathlib import Path


class AdaRank(sklearn.base.BaseEstimator):
    """AdaRank algorithm"""

    def __init__(self, max_iter=500, tol=0.0001, verbose=False, scorer=None):
        self.max_iter = max_iter
        self.n_iter = 0
        self.tol = tol
        self.verbose = verbose
        self.scorer = scorer
        self.coef_ = None

    def fit(self, x, y, qid, x_valid=None, y_valid=None, qid_valid=None):
        """Fit a model to the data"""
        x, y = check_X_y(x, y, 'csr')

        if x_valid is None:
            x_valid, y_valid, qid_valid = x, y, qid
        else:
            x_valid, y_valid = check_X_y(x_valid, y_valid, 'csr')

        n_queries = np.unique(qid).shape[0]
        weights = np.ones(n_queries, dtype=np.float64) / n_queries
        weak_rankers = []
        coef = np.zeros(x.shape[1])

        # use nCSDCG@10 as the default scorer
        if self.scorer is None:
            self.scorer = NCSDCGScorerQid(k=10)

        # precompute performance measurements for all weak rankers
        weak_ranker_score = []
        for j in range(x.shape[1]):
            pred = x[:, j].ravel()
            weak_ranker_score.append(self.scorer(y, pred, qid, x))

        best_perf_train = -np.inf
        best_perf_valid = -np.inf
        used_fids = []

        while self.n_iter < self.max_iter:
            self.n_iter += 1
            best_weighted_average = -np.inf
            best_weak_ranker = None
            for fid, score in enumerate(weak_ranker_score):
                if fid in used_fids:
                    continue
                weighted_average = np.dot(weights, score)
                if weighted_average > best_weighted_average:
                    best_weak_ranker = {'fid': fid, 'score': score}
                    best_weighted_average = weighted_average

            # stop when all the weaker rankers are out
            if best_weak_ranker is None:
                print("there are no weak rankers")
                break

            h = best_weak_ranker
            h['alpha'] = 0.5 * (math.log(np.dot(weights, 1 + h['score']) /
                                         np.dot(weights, 1 - h['score'])))
            weak_rankers.append(h)

            # update the ranker
            coef[h['fid']] += h['alpha']

            # score both training and validation data
            scorer_args = (y, np.dot(x, coef), qid, x)
            vali_scorer_args = (y_valid, np.dot(x_valid, coef), qid_valid, x_valid)
            score_train = self.scorer(*scorer_args)
            perf_train = score_train.mean()

            perf_valid = perf_train
            if x_valid is not x:
                perf_valid = self.scorer(*vali_scorer_args).mean()

            if self.verbose:
                print('{n_iter}\t{alpha}\t{fid}\t{score}\ttrain {train:.4f}\tvalid {valid:.4f}'.
                      format(n_iter=self.n_iter, alpha=h['alpha'], fid=h['fid'],
                             score=h['score'][:5], train=perf_train, valid=perf_valid),
                      file=sys.stderr)

            # update the best validation scores
            if perf_valid > best_perf_valid + self.tol:
                best_perf_valid = perf_valid
                self.coef_ = coef.copy()

            # update the best training score
            if perf_train > best_perf_train + self.tol:
                best_perf_train = perf_train

            # update weights
            new_weights = np.exp(-score_train)
            weights = new_weights / new_weights.sum()
        return self

    def predict(self, x):
        """Make predictions"""
        return np.dot(x, self.coef_)

    def save_weights(self, filename=None):
        """Save weights"""
        save_path_directory = Path(__file__).parent.joinpath("model_weights")
        save_path_directory.mkdir(parents=True, exist_ok=True)
        weights_path = (Path(save_path_directory).joinpath("redorank_weights.npy") if not filename
                        else Path(save_path_directory).joinpath(f"{filename}"))
        np.save(weights_path, self.coef_)

    def load_weights(self, filename=None):
        """Load weights"""
        weights_directory = Path(__file__).parent.joinpath("model_weights")
        weights_path = (Path(weights_directory).joinpath("redorank_weights.npy") if not filename
                        else Path(weights_directory).joinpath(f"{filename}"))
        self.coef_ = np.load(weights_path)
