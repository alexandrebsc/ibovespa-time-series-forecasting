"""Script for the module TimeSeriesSplitter."""

import numpy as np
import pandas as pd

from utils.constants import Col


class TimeSeriesSplitter:
    """Module responsible to splitting a time series for test and training."""

    def __init__(self, test_days: int) -> None:
        """Initialize a TimeSeriesSplitter with a specific amount of test days."""
        self.test_days = test_days

    def split(
        self, df_raw: pd.DataFrame, df_processed: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a time series into test and train Dataframes.

        :param df_raw: Dataframe used to generate test data
        :type df_raw: pd.DataFrame
        :param df_processed: Dataframe used to generate train data
        :type df_processed: pd.DataFrame
        :return: Train Dataframe, Test Dataframe
        :rtype: tuple[DataFrame, DataFrame]
        """
        def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.reset_index()
                .rename(columns={Col.date: Col.ds, Col.close: Col.y})
                .set_index(Col.ds)
            )

        df_test = rename_columns(df_raw).sort_index().tail(self.test_days)
        df_train = rename_columns(df_processed).sort_index().drop(df_test.index)

        df_train[Col.y] = df_train[Col.y].astype(np.float32)
        df_test[Col.y] = df_test[Col.y].astype(np.float32)

        return df_train.reset_index(), df_test.reset_index()
