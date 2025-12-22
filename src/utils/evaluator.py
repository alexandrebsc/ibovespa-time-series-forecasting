"""Script for the module Evaluator."""

# ruff: noqa: G004

import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    """Module responsible for evaluating the model predictions."""

    def __init__(self) -> None:
        """Initialize an Evaluator."""
        self.log = logging.getLogger(__name__)

    def evaluate(
        self,
        y_train: NDArray,
        predicted_train_values: NDArray,
        y_validate: NDArray,
        predicted_validation_values: NDArray,
    ) -> None:
        """Evaluate a models prediction.

        :param y_train: Model Y values for training
        :type y_train: NDArray
        :param y_validate: Model Y values for validate
        :type y_validate: NDArray
        :param predicted_train_values: Model predicted values for training
        :type predicted_train_values: NDArray
        :param predicted_validation_values: Model predicted values for validation
        :type predicted_validation_values: NDArray
        """
        self.y_train = y_train
        self.predicted_train_values = predicted_train_values
        self.y_validate = y_validate
        self.predicted_validation_values = predicted_validation_values

        self.performance_metrics_info(
            y_validate,
            predicted_validation_values,
            "LSTM Model",
        )
        self._plot_predictions_for_validation_and_train()

        self.evaluate_naive_persistence_baseline(y_train, y_validate)

    def evaluate_naive_persistence_baseline(
        self,
        y_train: NDArray,
        y_validate: NDArray,
    ) -> None:
        """Evaluate a naive persistence baseline model.

        This model predicts the same price as the previous day.

        :param y_train: Actual values for training
        :type y_train: NDArray
        :param y_validate: Actual values for validation
        :type y_validate: NDArray
        """
        self.log.info("Naive Persistence Baseline Model Evaluation")

        naive_train_predictions = np.zeros_like(y_train)
        naive_train_predictions[0] = y_train[0]
        for i in range(1, len(y_train)):
            naive_train_predictions[i] = y_train[i - 1]

        naive_validation_predictions = np.zeros_like(y_validate)
        if len(y_train) > 0:
            naive_validation_predictions[0] = y_train[-1]
        for i in range(1, len(y_validate)):
            naive_validation_predictions[i] = y_validate[i - 1]

        return self.performance_metrics_info(
            y_actual=y_validate,
            y_predicted=naive_validation_predictions,
            model_name="Naive Persistence Baseline",
            print_info=True,
        )

    def _plot_predictions_for_validation_and_train(self) -> None:
        """Plot the model predictions for validation and training."""
        fig, ax = plt.subplots(2, 1, figsize=(16, 7))

        ax[0].set_title("Previsões LSTM para dados de validação")
        ax[0].plot(self.y_validate, ".b", label="Real não conhecido")
        ax[0].plot(self.predicted_validation_values, ".r", label="Previsto")
        ax[0].legend()

        ax[1].set_title("Previsões LSTM para dados de treinamento")
        ax[1].plot(self.y_train, label="Real usado no treinamento")
        ax[1].plot(self.predicted_train_values, label="Previsto")
        ax[1].legend()

        fig.tight_layout()
        plt.show()

    def _plot_predictions_for_validation(self) -> None:
        """Plot the model predictions for validation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title("Previsões LSTM para dados de validação")
        ax.plot(self.y_validate, ".b", label="Real não conhecido")
        ax.plot(self.predicted_validation_values, ".r", label="Previsto")
        ax.legend()

        fig.tight_layout()
        plt.show()

    def _get_train_mse(self) -> float:
        """Get the training MSE.

        :return: The training MSE
        :rtype: float
        """
        return mean_squared_error(self.y_train, self.predicted_train_values)

    def performance_metrics_info(
        self,
        y_actual: NDArray,
        y_predicted: NDArray,
        model_name: str | None = None,
        *,
        print_info: bool = True,
    ) -> tuple[float, float, float, float]:
        """Get performance metrics for the model.

        :param y_actual: Actual values (optional, uses stored values if None)
        :type y_actual: NDArray | None
        :param y_predicted: Predicted values (optional, uses stored values if None)
        :type y_predicted: NDArray | None
        :param model_name: Name of the model for display purposes
        :type model_name: str
        :param print_info: If True, print the metrics
        :type print_info: bool
        :return: In order return: MSE, MAE, RMSE, R2
        :rtype: tuple[float, float, float, float]
        """
        mse = mean_squared_error(y_actual, y_predicted)
        mae = mean_absolute_error(y_actual, y_predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_predicted)

        if print_info:
            self.log.info(f"\n{model_name} Performance Metrics:")
            self.log.info(f"Mean Squared Error (MSE): {mse:.6f}")
            self.log.info(f"Mean Absolute Error (MAE): {mae:.6f}")
            self.log.info(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
            self.log.info(f"R-squared (R2) Score: {r2:.6f}")

        return mse, mae, rmse, r2
