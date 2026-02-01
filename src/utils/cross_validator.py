"""Script for cross validating the model."""

import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd

from utils.constants import Col
from utils.evaluator import Evaluator
from utils.ibovespa_loader import IbovespaLoader
from utils.ibovespa_preprocessor import IbovespaPreprocessor
from utils.lstm_model import LSTMModel, TrainingConfig


class CrossValidator:
    """Module for cross-validation."""

    def __init__(self) -> None:
        """Initialize a CrossValidationRunner."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CrossValidationRunner...")

        self.df_raw = IbovespaLoader().load()
        self.df_processed = IbovespaPreprocessor().preprocess(
            self.df_raw,
            adjust_for_inflation=True,
        )
        self.logger.info("Data loaded and preprocessed successfully")

        self.evaluator = Evaluator()

    def run(
        self,
        training_config: TrainingConfig,
        validation_days: int,
    ) -> list[dict[str, Any]]:
        """Run cross validation and return performance metrics."""
        self.logger.info("Starting cross-validation...")
        folds_performance_metrics = []

        for fold in self._get_k_folds_dicts(
            sequence_length=training_config.sequence_length,
            validation_days=validation_days,
            number_of_cross_validations=5,
        ):
            self.logger.info("Processing: %s", fold["info"])

            model = LSTMModel(training_config)
            _, _, y_validate, predicted_validation_values = model.fit(
                fold["df_train"],
                fold["df_validate"],
            )

            mse, mae, rmse, r2 = self.evaluator.performance_metrics_info(
                y_validate,
                predicted_validation_values,
                print_info=False,
            )

            self.logger.debug(
                "Fold metrics - MSE: %.4f, MAE: %.4f, RMSE: %.4f, R²: %.4f",
                mse,
                mae,
                rmse,
                r2,
            )

            folds_performance_metrics.append(
                {"info": fold["info"], "mse": mse, "mae": mae, "rmse": rmse, "r2": r2},
            )

        self._log_fold_metrics(folds_performance_metrics)
        self._log_average_metrics(folds_performance_metrics)

        return folds_performance_metrics

    def _log_fold_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Log individual fold metrics."""
        self.logger.info("\n" + "=" * 60)  # noqa: G003
        self.logger.info("INDIVIDUAL FOLD METRICS")
        self.logger.info("=" * 60)

        for performance in metrics:
            self.logger.info("\n%s", performance["info"])
            self.logger.info("  MSE:  %.6f", performance["mse"])
            self.logger.info("  MAE:  %.6f", performance["mae"])
            self.logger.info("  RMSE: %.6f", performance["rmse"])
            self.logger.info("  R²:   %.6f", performance["r2"])

    def _log_average_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Log average cross-validation metrics."""
        avg_mse = np.mean([m["mse"] for m in metrics])
        avg_mae = np.mean([m["mae"] for m in metrics])
        avg_rmse = np.mean([m["rmse"] for m in metrics])
        avg_r2 = np.mean([m["r2"] for m in metrics])

        self.logger.info("\n" + "=" * 60)  # noqa: G003
        self.logger.info("CROSS-VALIDATION AVERAGE METRICS")
        self.logger.info("=" * 60)
        self.logger.info("Average MSE:  %.6f", avg_mse)
        self.logger.info("Average MAE:  %.6f", avg_mae)
        self.logger.info("Average RMSE: %.6f", avg_rmse)
        self.logger.info("Average R²:   %.6f", avg_r2)

        std_mse = np.std([m["mse"] for m in metrics])
        std_mae = np.std([m["mae"] for m in metrics])
        std_r2 = np.std([m["r2"] for m in metrics])

        self.logger.info("\nStandard Deviations:")
        self.logger.info("  MSE: ±%.6f", std_mse)
        self.logger.info("  MAE: ±%.6f", std_mae)
        self.logger.info("  R²:  ±%.6f", std_r2)

    def _get_k_folds_dicts(
        self,
        validation_days: int,
        sequence_length: int,
        number_of_cross_validations: int = 5,
    ) -> Generator[dict[str, Any], None, None]:
        """Generate K-fold cross-validation splits."""
        df_train_complete, df_validation_complete = (
            self._get_complete_train_and_validation_df()
        )

        for i in range(number_of_cross_validations):
            df_validation = df_validation_complete.tail(
                sequence_length + validation_days,
            )
            df_train = df_train_complete.drop(df_validation.tail(validation_days).index)

            train_date_range = (
                f"{df_train.index.min().date()} - {df_train.index.max().date()}"
            )

            validation_date_range = (
                f"{df_validation.head(sequence_length + 1).index.max().date()} "
                f"- {df_validation.index.max().date()}"
            )
            yield {
                "df_validate": df_validation.copy(),
                "df_train": df_train.copy(),
                "info": (
                    f"Fold {i + 1}: "
                    f"Train [{train_date_range}] | "
                    f"Validation [{validation_date_range}]"
                ),
            }

            df_validation_complete = df_validation_complete.drop(
                df_validation.tail(validation_days).index,
            )
            df_train_complete = df_train_complete.drop(
                df_validation.tail(validation_days).index,
            )

    def _get_complete_train_and_validation_df(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare train and validation dataframes."""

        def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.reset_index()
                .rename(columns={Col.date: Col.ds, Col.close: Col.y})
                .set_index(Col.ds)
            )

        df_validation_complete = rename_columns(self.df_raw).sort_index()
        df_train_complete = rename_columns(self.df_processed).sort_index()

        df_train_complete[Col.y] = df_train_complete[Col.y].astype(np.float32)
        df_validation_complete[Col.y] = df_validation_complete[Col.y].astype(np.float32)

        self.logger.debug("Training data shape: %s", df_train_complete.shape)
        self.logger.debug("Validation data shape: %s", df_validation_complete.shape)

        return df_train_complete, df_validation_complete
