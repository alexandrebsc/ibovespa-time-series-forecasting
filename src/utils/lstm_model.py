"""Script for the module LSTMModel."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from utils.constants import Col


@dataclass
class TrainingConfig:
    """Configuration for LSTM model training."""

    sequence_length: int
    neurons: int
    epochs: int
    batch_size: int


class LSTMModel:
    """Module responsible for training a LSTM model."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize a configured LSTMModel."""
        self.log = logging.getLogger(__name__)

        self.sequence_length = config.sequence_length
        self.neurons = config.neurons
        self.epochs = config.epochs
        self.batch_size = config.batch_size

    def fit(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Train and test the model.

        :param df_train: Training Dataframe
        :type df_train: pd.DataFrame
        :param df_test: Test Dataframe
        :type df_test: pd.DataFrame
        :return: In the following order: y_train, predicted_train_values, y_validate and predicted_validation_values
        :rtype: tuple[NDArray, NDArray, NDArray, NDArray]
        """  # noqa: E501
        x_train, y_train = self._create_input_lstm_sequences(
            df_train[Col.y].values,
            df_train[Col.y].values,
        )
        x_validate, y_validate = self._create_input_lstm_sequences(
            df_test[Col.y].values,
            df_test[Col.y].values,
        )

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_validate = x_validate.reshape((x_validate.shape[0], x_validate.shape[1], 1))

        self.model = Sequential(
            [
                LSTM(
                    units=self.neurons,
                    activation="relu",
                    input_shape=(self.sequence_length, 1),
                ),
                Dense(units=1),
            ],
        )

        self.model.compile(optimizer="adam", loss="mean_squared_error")

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_validate, y_validate),
        )

        return (
            y_train,
            self.model.predict(x_train),
            y_validate,
            self.model.predict(x_validate),
        )

    def get_model(self) -> Sequential:
        """Return the model. Must fit() first."""
        return self.model

    def save_model(self) -> None:
        """Save the model."""
        self.model.save("artifacts/model.keras")

    def _create_input_lstm_sequences(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        sequences = [
            (x[i : i + self.sequence_length], y[i + self.sequence_length])
            for i in range(len(x) - self.sequence_length)
        ]

        return (
            np.array([seq[0] for seq in sequences]),
            np.array(
                [seq[1] for seq in sequences],
            ),
        )
