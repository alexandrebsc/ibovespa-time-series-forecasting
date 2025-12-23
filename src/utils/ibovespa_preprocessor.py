"""Script for the module IbovespaPreprocessor."""

# ruff: noqa: DTZ001, G004
import logging
from datetime import datetime

import pandas as pd

from utils.constants import IPCA_HIST_CSV_PATH, Col


class IbovespaPreprocessor:
    """Module responsible to preprocess the Ibovespa time series data usable for modeling."""  # noqa: E501

    def __init__(self) -> None:
        """Initialize a IbovespaPreprocessor."""
        self.log = logging.getLogger(__name__)

    def preprocess(
        self,
        df: pd.DataFrame,
        *,
        remove_outliers: bool = False,
        adjust_for_inflation: bool = False,
    ) -> pd.DataFrame:
        """Preprocess Ibovespa time series for financial modeling.

        :param df: Ibovespa time series
        :type df: pd.DataFrame
        :param remove_outliers: If True, remove 2008 financial crisis and COVID-19 pandemic periods
        :type remove_outliers: bool
        :param adjust_for_inflation: If True, adjust prices to real terms using IPCA inflation index
        :type adjust_for_inflation: bool
        :return: Preprocessed Ibovespa DataFrame ready for modeling
        :rtype: DataFrame
        """  # noqa: E501
        df = self._null_filling(df)
        df = self._duplicated_fix(df)
        df = df.drop([Col.max, Col.min, Col.open, Col.vol], axis=1)

        if remove_outliers:
            df = self._remove_outliers()

        if adjust_for_inflation:
            df = self._adjust_for_inflation(df)

        return df

    def _duplicated_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = [Col.close, Col.open, Col.max, Col.min]

        duplicated_mask = df.duplicated(subset=price_cols, keep="first")
        n_duplicated = duplicated_mask.sum()

        if n_duplicated > 0:
            dropped_dates = df.index[duplicated_mask]
            self.log.warning(
                f"{n_duplicated} duplicated rows detected based on "
                f"{price_cols}. Dropping occurrences: "
                f"{dropped_dates.tolist()}",
            )

        return df.loc[~duplicated_mask]

    def _null_filling(self, df: pd.DataFrame) -> pd.DataFrame:
        null_mask = df.isna()
        n_nulls = null_mask.sum().sum()

        if n_nulls == 0:
            return df

        self.log.warning(
            f"{n_nulls} missing values detected. "
            "Filling with rolling mean of previous 5 days.",
        )

        rolling_mean = df.shift(1).rolling(window=5, min_periods=1).mean()

        return df.where(~null_mask, rolling_mean)

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        crise_2008_start_2 = "2008-06-30"
        crise_2008_end_2 = "2008-07-14"

        crise_2008_start_1 = "2008-10-01"
        crise_2008_end_1 = "2009-04-24"

        covid_start = "2020-03-01"
        covid_end = "2020-07-18"

        return df[
            ~((df.index >= crise_2008_start_1) & (df.index <= crise_2008_end_1))
            & ~((df.index >= crise_2008_start_2) & (df.index <= crise_2008_end_2))
            & ~((df.index >= covid_start) & (df.index <= covid_end))
        ]

    def _adjust_for_inflation(self, df: pd.DataFrame) -> pd.DataFrame:
        ipca_df = self.__get_ipca_df()
        ipca_df = self.__fill_ipca_nulls(ipca_df, df).sort_index()
        ipca_df[Col.acc_inflation] = (1 + ipca_df[Col.inflation][::-1]).cumprod()[
            ::-1
        ] - 1

        end_month_shift_for_resample = ipca_df.index.max() + pd.offsets.DateOffset(
            months=1,
        )
        ipca_df.loc[end_month_shift_for_resample] = [0, 0]
        ipca_df = ipca_df.resample("D").ffill().loc[df.index]

        df[Col.close] = df[Col.close] * (ipca_df[Col.acc_inflation] + 1)
        df[Col.close] = df[Col.close].round().astype("int64")

        return df

    def __written_month_to_datetime(self, value: str) -> datetime:
        value = value.lower().strip()
        month_str, year = value.rsplit(" ", 1)

        hash_mes = {
            "janeiro": 1,
            "fevereiro": 2,
            "março": 3,
            "abril": 4,
            "maio": 5,
            "junho": 6,
            "julho": 7,
            "agosto": 8,
            "setembro": 9,
            "outubro": 10,
            "novembro": 11,
            "dezembro": 12,
        }

        return datetime(int(year), hash_mes[month_str], 1)

    def __get_ipca_df(self) -> pd.DataFrame:
        load_df = pd.read_csv(
            IPCA_HIST_CSV_PATH,
            sep=";",
            skiprows=1,
            skipfooter=1,
            engine="python",
        ).melt(var_name=Col.month, value_name=Col.inflation)

        load_df = load_df[
            pd.to_numeric(load_df[Col.inflation], errors="coerce").notna()
        ]
        load_df[Col.month] = load_df[Col.month].apply(self.__written_month_to_datetime)
        load_df[Col.inflation] = load_df[Col.inflation] / 100
        return load_df.set_index(Col.month)

    def __fill_ipca_nulls(
        self,
        df_inflation: pd.DataFrame,
        df_ibovespa: pd.DataFrame,
    ) -> pd.DataFrame:
        full_index = df_ibovespa.index.map(lambda x: x.replace(day=1)).unique()

        df_inflation = df_inflation.reindex(full_index)
        df_inflation = df_inflation.rename_axis(Col.month)

        null_found = df_inflation.isna().sum().iloc[0]
        if null_found:
            self.log.warning(
                f"{null_found} months without inflation values. "
                "Zero will be assumed for these.",
            )
            df_inflation = df_inflation.fillna(0)

        return df_inflation
