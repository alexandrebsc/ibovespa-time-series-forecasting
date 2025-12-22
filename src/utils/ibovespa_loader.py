"""Script for the module IbovespaLoader."""

from datetime import datetime

import pandas as pd

from utils.constants import IBOV_HIST_CSV_PATH, Col


class IbovespaLoader:
    """Module responsible to make the Ibovespa time series data readable."""

    def __init__(self) -> None:
        """Initialize a IbovespaLoader."""

    def load(self) -> pd.DataFrame:
        """Load the Ibovespa time series.

        :return: Ibovespa time series
        :rtype: DataFrame
        """
        return (
            pd.read_csv(
                IBOV_HIST_CSV_PATH,
                encoding="utf-8-sig",
                header=0,
                usecols=[0, 1, 2, 3, 4, 5],
                converters={
                    0: self._dot_formated_time_to_datetime,
                    1: self._dot_thousands_to_int,
                    2: self._dot_thousands_to_int,
                    3: self._dot_thousands_to_int,
                    4: self._dot_thousands_to_int,
                    5: self._vol_to_number,
                },
                names=[Col.date, Col.close, Col.open, Col.max, Col.min, Col.vol],
                engine="python",
            )
            .set_index(Col.date)
            .convert_dtypes()
            .sort_index()
        )

    def _dot_thousands_to_int(self, dot_thousand: str) -> int:
        return int(dot_thousand.replace(".", ""))

    def _dot_formated_time_to_datetime(self, dot_formated_time: str) -> datetime:
        return datetime.strptime(dot_formated_time, "%d.%m.%Y")  # noqa: DTZ007

    def _vol_to_number(self, str_number: str) -> int:
        if len(str_number) < 1:
            return None
        multiplier = 0
        if str_number[-1] == "K":
            multiplier = 10e2
        if str_number[-1] == "M":
            multiplier = 10e5
        return int(float(str_number[:-1].replace(",", ".")) * multiplier)
