"""Script for the module ForecastingPipeline."""

import random

import numpy as np
import tensorflow as tf

from utils.evaluator import Evaluator
from utils.ibovespa_loader import IbovespaLoader
from utils.ibovespa_preprocessor import IbovespaPreprocessor
from utils.lstm_model import LSTMModel
from utils.time_series_splitter import TimeSeriesSplitter


class ForecastingPipeline:
    """Pipeline for the complete ML workflow."""

    def __init__(  # noqa: PLR0913
        self,
        loader: IbovespaLoader,
        preprocessor: IbovespaPreprocessor,
        splitter: TimeSeriesSplitter,
        model: LSTMModel,
        evaluator: Evaluator = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a ForecastingPipeline."""
        self.loader = loader
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.model = model
        self.evaluator = evaluator
        self.random_seed = random_seed

    def run(self) -> None:
        """Pipeline that load data, preprocess it, split in training and test datasets, train the model and evaluate it."""  # noqa: E501
        self.__set_random_seeds()

        df_raw = self.loader.load()
        df_processed = self.preprocessor.preprocess(df_raw.copy())
        df_train, df_test = self.splitter.split(
            df_raw=df_raw,
            df_processed=df_processed,
        )
        self.predictions_and_y = self.model.fit(df_train, df_test)

        if self.evaluator is not None:
            self.evaluator.evaluate(*self.predictions_and_y)

    def __set_random_seeds(self) -> None:
        if self.random_seed is None:
            return

        np.random.default_rng(self.random_seed)
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
