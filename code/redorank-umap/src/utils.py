"""
All methods prior to the ReDOrank comment come from https://github.com/rueycheng/AdaRank/blob/master/utils.py

REdORank file reading inspired by the implementation found here:
https://github.com/bing0n3/AdaRank-Python/blob/master/read_data.py
"""
from sklearn.datasets import load_svmlight_file
from pathlib import Path
import numpy as np
import pandas as pd
import re
import sys
import logging
LOGGER = logging.getLogger(__name__)


def group_counts(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    return np.diff(np.where(np.append(d, 1))[0])


def group_offsets(arr):
    """Return a sequence of start/end offsets for the value subgroups in the input"""
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    idx = np.where(np.append(d, 1))[0]
    return zip(idx, idx[1:])


def load_docno(fname, letor=False):
    """Load docnos from the input in the SVMLight format"""
    if letor:
        docno_pattern = re.compile(r'#\s*docid\s*=\s*(\S+)')
    else:
        docno_pattern = re.compile(r'#\s*(\S+)')

    docno = []
    for line in open(fname):
        if line.startswith('#'):
            continue
        m = re.search(docno_pattern, line)
        if m is not None:
            docno.append(m.group(1))
    return np.array(docno)


def print_trec_run(qid, docno, pred, run_id='exp', output=None):
    """Print TREC-format run to output"""
    if output is None:
        output = sys.stdout
    for a, b in group_offsets(qid):
        idx = np.argsort(-pred[a:b]) + a  # note the minus and plus a
        for rank, i in enumerate(idx, 1):
            output.write('{qid} Q0 {docno} {rank} {sim} {run_id}\n'.
                         format(qid=qid[i], docno=docno[i], rank=rank, sim=pred[i], run_id=run_id))


class ReadData:
    def __init__(self):
        self.test_fold = []
        self.valid_fold = []
        self.train_fold = []
        self.data_frame = None

    def read_newsela(self, path, model_name) -> None:
        x_train, y_train, qid_train = load_svmlight_file(str(Path(path).joinpath(f"{model_name}_train_data.txt")), query_id=True)
        x_test, y_test, qid_test = load_svmlight_file(str(Path(path).joinpath(f"{model_name}_test_data.txt")), query_id=True)
        x_val, y_val, qid_val = load_svmlight_file(str(Path(path).joinpath(f"{model_name}_val_data.txt")), query_id=True)

        x_train = x_train.toarray()
        x_test = x_test.toarray()
        x_val = x_val.toarray()
        self.train_fold.append((x_train, y_train, qid_train))
        self.test_fold.append((x_test, y_test, qid_test))
        self.valid_fold.append((x_val, y_val, qid_val))

    def get_fold(self, n):
        return self.train_fold[n], self.test_fold[n], self.valid_fold[n]

    def read_dataset(self, dataset_path: Path, dataset_name: str, store_dataframe: bool = True) -> None:
        # Define the full data path
        if "kid-friend" in dataset_name:
            # There are three possibilities: Google, Bing, BM25
            if "google" in dataset_name:
                data_path = Path(dataset_path).joinpath(f"{dataset_name.replace('_google', '')}",
                                            "all_features", f"{dataset_name}_test_data.txt")
            elif "bing" in dataset_name:
                data_path = Path(dataset_path).joinpath(f"{dataset_name.replace('_bing', '')}",
                                            "all_features", f"{dataset_name}_test_data.txt")
            else:
                # BM25 is the default
                data_path = Path(dataset_path).joinpath(f"{dataset_name.replace('_bm25', '')}",
                                            "all_features", f"{dataset_name}_test_data.txt")

        elif dataset_name == "requik":
            data_path = Path(dataset_path).joinpath(f"{dataset_name}", "all_features", f"{dataset_name}_test_data.txt")
        else:
            LOGGER.error(f"\tDataset {dataset_name} is not supported.")

        # Load and pre-process the data
        x_test = pd.read_csv(str(Path(data_path)),
                             names=["rank", "qid", "edu_prob", "readability_level", "obj_prob", "doc_id"], sep=" ")
        x_test["qid"] = x_test["qid"].apply(lambda x: int(x.split(":")[-1]))
        x_test["edu_prob"] = x_test["edu_prob"].apply(lambda x: x.split(":")[-1])
        x_test["edu_prob"] = pd.to_numeric(x_test["edu_prob"])
        x_test["obj_prob"] = x_test["obj_prob"].apply(lambda x: x.split(":")[-1])
        x_test["obj_prob"] = pd.to_numeric(x_test["obj_prob"])
        x_test["readability_level"] = x_test["readability_level"].apply(lambda x: x.split(":")[-1])
        x_test["readability_level"] = pd.to_numeric(x_test["readability_level"])
        x_test["doc_id"] = x_test["doc_id"].apply(lambda x: x.split(":")[-1])
        y_test = np.array(x_test["rank"].tolist(), dtype=np.float64)
        qid_test = np.array(x_test["qid"].tolist(), dtype=np.float64)
        self.test_fold.append((x_test[["edu_prob", "readability_level", "obj_prob"]].to_numpy(), y_test, qid_test))
        # This is to allow get_fold to work, but users should ignore train and valid folds
        self.valid_fold.append(())
        self.train_fold.append(())

        if store_dataframe:
            self.data_frame = x_test

    def get_data_frame(self):
        return self.data_frame.copy()