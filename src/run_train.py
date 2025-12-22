"""Script for training the chosen model: LSTM."""

# ruff: noqa: D103
import logging

from utils.config import OPTIMAL_TEST_DAYS, OPTIMAL_TRAINING_CONIFG
from utils.constants import RANDOM_SEED
from utils.evaluator import Evaluator
from utils.forecasting_pipeline import ForecastingPipeline
from utils.ibovespa_loader import IbovespaLoader
from utils.ibovespa_preprocessor import IbovespaPreprocessor
from utils.lstm_model import LSTMModel
from utils.time_series_splitter import TimeSeriesSplitter


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    ForecastingPipeline(
        loader=IbovespaLoader(),
        preprocessor=IbovespaPreprocessor(),
        splitter=TimeSeriesSplitter(test_days=OPTIMAL_TEST_DAYS),
        model=LSTMModel(OPTIMAL_TRAINING_CONIFG),
        evaluator=Evaluator(),
        random_seed=RANDOM_SEED,
    ).run()


if __name__ == "__main__":
    main()
