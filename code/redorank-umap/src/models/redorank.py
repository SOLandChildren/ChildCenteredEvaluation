import logging
from numpy.typing import ArrayLike
from src.metrics import NCSDCGScorerQid
from src.models.adarank import AdaRank
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class REdORank(AdaRank):
    def __init__(self, max_iter: int = 500, tol: float = 0.0001, verbose: bool = False,
                 scorer: object = NCSDCGScorerQid(k=30), load_weights: bool = True) -> None:
        super().__init__(max_iter=max_iter, tol=tol, verbose=verbose, scorer=scorer)
        self._data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "ranking", "data")

        if load_weights:
            self._load_weights()

    def train(self, training_features: ArrayLike, train_targets: ArrayLike, train_qids: ArrayLike,
              validation_features: ArrayLike, validation_targets: ArrayLike, validation_qids: ArrayLike,
              save_weights: bool = True, weights_file: Path = None) -> None:
        """
            Method to train the model.

        :param training_features:
        :param train_targets:
        :param train_qids:
        :param validation_features:
        :param validation_targets:
        :param validation_qids:
        :param save_weights:
        :param weights_file:
        :return:
        """
        if save_weights:
            try:
                assert weights_file is not None
            except AssertionError:
                LOGGER.error("\tIf save_weights is True, a path must be given in weights_file")
        LOGGER.info("\tBeginning training...")
        self.fit(training_features, train_targets, train_qids, validation_features, validation_targets,
                        validation_qids)
        LOGGER.info("\tTraining complete!")
        if save_weights:
            LOGGER.info(f"\tSaving weights to {weights_file}...")
            self.save_weights(weights_file)
            LOGGER.info("\tWeights saved!")

    def set_data_path(self, data_path: Path) -> None:
        self._data_path = data_path

    def _load_weights(self, weights_file: Path = None) -> None:
        if weights_file is not None:
            self.load_weights(weights_file)
        else:
            weights_path = Path(__file__).resolve().parent.parent.joinpath(
                "model_weights", "redorank_weights.npy")
            self.load_weights(weights_path)

    def predict(self, x: ArrayLike) -> ArrayLike:
        """
            Performs the prediction step of the base AdaRank model.

        :param x: Features to perform prediction on
        :return: ndarray of predictions
        """
        return super().predict(x)
