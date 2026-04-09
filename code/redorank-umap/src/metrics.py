import numpy as np
from utils import group_offsets


class Scorer(object):
    def __init__(self, score_func, **kwargs):
        self.score_func = score_func
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.score_func(*args, **self.kwargs)


# Precision
#
def _p_score(y_true, y_pred, k=None):
    order = np.argsort(-y_pred)
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true > 0) / len(y_true)


def p_score(y_true, y_pred, qid, k=None):
    return np.array([_p_score(y_true[a:b], y_pred[a:b], k=k) for a, b in group_offsets(qid)])


class PScorer(Scorer):
    def __init__(self, **kwargs):
        super(PScorer, self).__init__(p_score, **kwargs)


# AP (Average Precision)
#
def _ap_score(y_true, y_pred):
    order = np.argsort(-y_pred)
    y_true = np.take(y_true, order)
    pos = 1 + np.where(y_true > 0)[0]
    n_rels = 1 + np.arange(len(pos))
    return np.mean(n_rels / pos) if len(pos) > 0 else 0


def ap_score(y_true, y_pred, qid):
    return np.array([_ap_score(y_true[a:b], y_pred[a:b]) for a, b in group_offsets(qid)])


class APScorer(Scorer):
    def __init__(self):
        super(APScorer, self).__init__(ap_score)


# DCG/nDCG (Normalized Discounted Cumulative Gain)
#
def _burges_dcg(y_true, y_pred, k=None):
    # print(y_pred)
    # order = np.argsort(y_pred)[::-1]
    order = np.argsort(-y_pred)
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(gain)) + 2)
    return np.sum(gain / discounts)


def _trec_dcg(y_true, y_pred, k=None):
    order = np.argsort(-y_pred)
    y_true = np.take(y_true, order[:k])
    gain = y_true
    discounts = np.log2(np.arange(len(gain)) + 2)
    return np.sum(gain / discounts)


def _dcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
    assert dcg_func is not None
    y_true = np.maximum(y_true, 0)
    return np.array([dcg_func(y_true[a:b], y_pred[a:b], k=k) for a, b in group_offsets(qid)])


def _ndcg_score(y_true, y_pred, qid, k=None, dcg_func=None):
    assert dcg_func is not None
    y_true = np.maximum(y_true, 0)
    dcg = _dcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)
    idcg = np.array([dcg_func(np.sort(y_true[a:b]), np.arange(0, b - a), k=k)
                     for a, b in group_offsets(qid)])
    assert (dcg <= idcg).all()
    idcg[idcg == 0] = 1
    return dcg / idcg


def dcg_score(y_true, y_pred, qid, k=None, version='burges'):
    assert version in ['burges', 'trec']
    dcg_func = _burges_dcg if version == 'burges' else _trec_dcg
    return _dcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)


def ndcg_score(y_true, y_pred, qid, k=None, version='burges'):
    assert version in ['burges', 'trec']
    dcg_func = _burges_dcg if version == 'burges' else _trec_dcg
    return _ndcg_score(y_true, y_pred, qid, k=k, dcg_func=dcg_func)


class DCGScorer(Scorer):
    def __init__(self, **kwargs):
        super(DCGScorer, self).__init__(dcg_score, **kwargs)


class NDCGScorer(Scorer):
    def __init__(self, **kwargs):
        super(NDCGScorer, self).__init__(ndcg_score, **kwargs)


"""
REdORank metrics are inspired by the implementation found here:
https://github.com/bing0n3/AdaRank-Python/blob/master/metric.py
"""
# EXPERIMENTAL
def cs_dcg_at_k(y_pred, k, cost):
    gain = np.exp2(y_pred[:k]) - 1
    discounts = np.log2(np.arange(len(y_pred[:k])) + 2)
    return np.sum((gain / discounts) - cost[:k])


def ncsdcg_at_k(y_true, y_predict, x, k, cost_index):
    label_indices = np.flip(np.argsort(y_predict))
    csdcg_labels = np.take(y_true, label_indices)

    sorted_true = np.flip(np.argsort(y_true))
    y_true_sorted = np.take(y_true, sorted_true)

    if cost_index:
        cost = x[label_indices, cost_index]
        cost_best = x[:, cost_index]
        cost_best = np.take(cost_best, sorted_true)
        cost_worst = np.flip(np.take(x[:, cost_index], sorted_true))

    else:
        cost, cost_best, cost_worst = np.array([0]), np.array([0]), np.array([0])

    wcsdcg = cs_dcg_at_k(np.flip(y_true_sorted), k, cost_worst)
    icsdcg = cs_dcg_at_k(y_true_sorted, k, cost_best)
    csdcg = cs_dcg_at_k(csdcg_labels, k, cost)

    ncsdcg = (csdcg - wcsdcg) / (icsdcg - wcsdcg)

    return ncsdcg


class NCSDCGScorer(object):
    def __init__(self, k=5, cost_index=None):
        self.k = k
        self.cost_index = cost_index

    def __call__(self, y_true, y_predict, x):
        return ncsdcg_at_k(y_true, y_predict, x, self.k, self.cost_index)


class NCSDCGScorerQid:
    def __init__(self, k=5, cost_index=None):
        self.scorer = NCSDCGScorer(k, cost_index)
        self.k = k

    def __call__(self, y_true, y_predict, qid, x):
        return self.get_scores(y_true, y_predict, qid, x)

    def get_scores(self, true_list, pred_l, qid, x):
        scores = []
        # get group off-set
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
            scores.append(self.scorer(true_list[start:end], pred_l[start:end], x[start:end, :]))

        return np.array(scores)
