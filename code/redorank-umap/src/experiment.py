import argparse
import pandas as pd
import numpy as np
import logging

from metrics import NDCGScorer, NCSDCGScorerQid
from utils import ReadData
from src.models.adarank import AdaRank
from src.models.redorank import REdORank
from pathlib import Path
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def load_data(data_directory, model_identifier, with_splits=True):
    """
    Function to load data for training or inference.

    Expected directory hierarchy is shown below
    data_directory
         |
          -- model_name
         |       |
         |       -- all_features

    :param data_directory: Directory within which the data lives
    :param model_identifier: Name of the model the data is being collected for, this is also a subdirectory name
    :param with_splits: Whether there is train, test, val data splits present. If this is false, it is assumed
                        only test data is present.
    :return: ndarray or tuple of ndarrays
    """
    full_data_path = Path(data_directory).joinpath(f"{model_identifier}", "all_features")
    data_reader = ReadData()
    if with_splits:
        data_reader.read_newsela(full_data_path, model_identifier)
        train_data, test_data, val_data = data_reader.get_fold(0)
        return train_data, test_data, val_data
    else:
        _, test_data, _ = data_reader.get_fold(0)
        return test_data


if __name__ == "__main__":
    """
    Example execution call: python experiment.py --model kid-friend_bm25 -l --store_predictions
    """

    parser = argparse.ArgumentParser(description="Experimental script for evaluating REdORank re-ranking algorithm.")
    parser.add_argument("--model", type=str, required=True,
                        choices=("redorank_2026", "requik", "kid-friend_bing", "kid-friend_google", "kid-friend_bm25", "korsce"),
                        help="Defines which data to model using the REdORank algorithm.")
    parser.add_argument("--dataset_path", type=str,
                        default=Path(__file__).resolve().parent.parent.joinpath("datasets", "ranking", "data"),
                        help="Path to the data storage location.")
    parser.add_argument("-s", dest="save_weights", action="store_true", default=False,
                        help="Save training weights for later loading.")
    parser.add_argument("-l", dest="load_weights", action="store_true", default=False,
                        help="Indicate that the model should load saved weights. If none found, exits with an error message")
    parser.add_argument("-t", dest="train", action="store_true", default=False,
                        help="Flag to indicate this is a training run.")
    parser.add_argument("-r", dest="reproduce_thesis", action="store_true", default=False,
                        help="Flag to trigger the reproducibility run of the original REdORank experiment.")
    parser.add_argument("--store_predictions", action="store_true", default=False,
                        help="Flag to indicate whether to save the predictions to disk.")
    args = parser.parse_args()

    # Argument sanity checks
    if args.train and args.load_weights:
        LOGGER.warning("\tLoading weights is not possible on a train run. Ignoring -l flag.")
        args.load_weights = False

    if args.train and not args.save_weights:
        LOGGER.warning("\tInitiating training run without saving the weights. For more predictable reproducibility consider passing the -s flag.")

    if args.model == "korsce":
        from src.models.korsce import run_korsce
        run_korsce(pd.DataFrame(), NDCGScorer(k=30))

    if args.reproduce_thesis:
        assert args.model == "redorank_2026"
        LOGGER.info("\tLoading data...")
        test_features, train_features, val_features = load_data(args.dataset_path, args.model, with_splits=True)
        x_test, y_test, qid_test = test_features
        x_train, y_train, qid_train = train_features
        x_val, y_val, qid_val = val_features
        results_frame = pd.DataFrame(x_test, columns=["edu_prob", "readability_level", "obj_prob"])
        results_frame["qid"] = qid_test
        results_frame["rank"] = y_test

        ranker = REdORank()
        weights_path = Path(__file__).resolve().parent.joinpath("model_weights", f"{args.model}_weights.npy")
        LOGGER.info("\tTraining the model...")
        ranker.train(x_train, y_train, qid_train, x_val, y_val, qid_val, save_weights=True, weights_file=weights_path)
    else:
        LOGGER.info("\tLoading data...")
        data_reader = ReadData()
        data_reader.read_dataset(args.dataset_path, args.model)
        _, test_data, _ = data_reader.get_fold(0)
        x_test, y_test, qid_test = test_data
        results_frame = data_reader.get_data_frame()

        LOGGER.info(f"\tRunning REdORank for {args.model}")
        ranker = REdORank()

    relevance_predictions = ranker.predict(x_test)
    results_frame["relevance_pred"] = relevance_predictions
    results_frame_with_rank_prediction = pd.DataFrame()
    unique_qids = np.unique(qid_test)
    for u in tqdm(unique_qids, desc=">> Creating sorted ranked lists for each query..."):
        sliced = pd.DataFrame(results_frame[results_frame["qid"] == u].copy())
        sliced = sliced.sort_values(by=["relevance_pred"], ascending=False)
        sliced_size = sliced.shape[0]
        sliced["rank_pred"] = np.arange(0, sliced_size)
        results_frame_with_rank_prediction = pd.concat([results_frame_with_rank_prediction, sliced], ignore_index=True)

    if args.store_predictions:
        results_path = Path(__file__).resolve().parent.parent.parent.joinpath("results", "kid-friend")
        if "doc_id" in results_frame_with_rank_prediction.columns:
            results_frame_with_rank_prediction[["qid", "doc_id", "rank_pred", "relevance_pred"]].to_csv(
                Path(results_path).joinpath(
                    f"{args.model.replace('_google', '').replace('_bing', '').replace('_bm25', '')}",
                    f"{args.model}_rank_predictions.txt"), index=False, sep=" ", header=False)
        else:
            results_frame_with_rank_prediction[["qid", "rank_pred", "relevance_pred"]].to_csv(
                Path(results_path).joinpath(
                    f"{args.model.replace('_google', '').replace('_bing', '').replace('_bm25', '')}",
                    f"{args.model}_rank_predictions.txt"), index=False, sep=" ", header=False)

    result_scorer = NDCGScorer(k=10)
    performance = result_scorer(y_test, relevance_predictions, qid_test)
    print(f"{args.model}\t", performance.mean())
