"""Script for exploratory data analysis(EDA)."""

# ruff: noqa: D103, G004

import logging
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.dates import YearLocator
from mplfinance.original_flavor import candlestick_ohlc
from scipy.stats import shapiro
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from utils.constants import Col
from utils.ibovespa_loader import IbovespaLoader
from utils.ibovespa_preprocessor import IbovespaPreprocessor

VARIATION_FOMATTER = ticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
PRICE_FORMATTER = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    raw_df = IbovespaLoader().load()
    processed_df = IbovespaPreprocessor().preprocess(raw_df, adjust_for_inflation=True)

    df_analysis(raw_df)

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": ".9"})
    df_vizulization(raw_df)

    plot_line_close_price(processed_df)


def df_analysis(df: pd.DataFrame) -> None:
    def shapiro_test() -> None:
        warnings.filterwarnings(
            "ignore",
            message="scipy.stats.shapiro: For N > 5000",
            category=UserWarning,
        )

        df[Col.var] = df[Col.close].pct_change() * 100.0
        stat, p = shapiro(df[Col.var].dropna())

        logger.info(f"Statistics (W): {stat:.4f}")
        logger.info(f"Value p: {p:.4f}")
        logger.info("")

    def adf_test() -> None:
        adf_result = adfuller(df[Col.close])

        logger.info(f"ADF Statistic: {adf_result[0]:.4f}")
        logger.info(f"Value p: {adf_result[1]:.4f}")
        logger.info("Critical values:")

        for key, value in adf_result[4].items():
            logger.info(f"    {key}: {value:.4f}")

    def null_analysis() -> None:
        df_nulls = df.isna()

        logger.info(f"Nulls quantity:\n{df_nulls.sum()}")
        if df_nulls[Col.vol].any():
            logger.info(f"Nulls in {Col.vol} column:\n{df[df_nulls[Col.vol]]}")
        logger.info("")

    def duplicated_analysis() -> None:
        df_duplicated = df.duplicated()

        logger.info(f"Duplicated quantity: {df_duplicated.sum()}")
        if df_duplicated.sum() > 0:
            logger.info(f"Duplicated rows:\n{df[df_duplicated]}")
        logger.info("")

    logger.info("=" * 60)
    logger.info("DATAFRAME ANALYSIS")
    logger.info("=" * 60)

    logger.info("\n> Info:")
    df.info()
    logger.info("\n")

    logger.info("> Data Sample (first 3 rows):")
    logger.info(f"\n{df.head(3)}\n")

    logger.info("> Descriptive Statistics:")
    logger.info(f"\n{df.describe()}\n")

    logger.info("> Null Analysis:")
    null_analysis()

    logger.info("> Duplicate Analysis:")
    duplicated_analysis()

    logger.info("> Shapiro-Wilk Test on Daily Percentage Variation:")
    logger.info("(Testing for normality of returns)")
    shapiro_test()

    logger.info("> Augmented Dickey-Fuller Test on Close Price:")
    logger.info("(Testing for stationarity)")
    adf_test()

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


def df_vizulization(df: pd.DataFrame) -> None:
    df[Col.var] = df[Col.close].pct_change() * 100.0

    plot_count_per_year(df)
    plot_candle_stick(df)
    plot_line_close_price(df)
    plot_average_close_price_per_year(df)
    plot_variables_correlation(df)
    plot_box_violin(df)
    plot_kde(df)
    plot_variation(df)
    plot_seasonal_decompose(df, Col.close, period=500, formatter=PRICE_FORMATTER)
    plot_seasonal_decompose(df, Col.var, period=500, formatter=VARIATION_FOMATTER)
    plot_outliers_detection(df)


def plot_candle_stick(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot = df_plot.reset_index()
    df_plot[Col.date] = df_plot[Col.date].apply(mdates.date2num)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim([0, 150000])

    ohlc = df_plot[[Col.date, Col.open, Col.max, Col.min, Col.close]].to_numpy()

    candlestick_ohlc(ax, ohlc, width=0.6, colorup="g", colordown="r")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(YearLocator())
    ax.yaxis.set_major_formatter(PRICE_FORMATTER)

    plt.xticks(rotation=45)

    plt.title("Gráfico Candlestick BVSP", weight="bold")
    plt.xlabel("Data")
    plt.ylabel("Preço")

    plt.tight_layout()
    plt.show()


def plot_count_per_year(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["year"] = df_plot.index.year
    counts_per_year = df_plot["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=counts_per_year.index, y=counts_per_year.values)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de dias de trade")
    ax.set_title("Número de dias de trade por ano", weight="bold")
    plt.xticks(rotation=45)

    fig.tight_layout()
    plt.show()


def plot_line_close_price(df: pd.DataFrame) -> None:
    df_plot = df.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x=Col.date, y=Col.close, data=df_plot)

    ax.set_xlabel("Data")
    ax.set_ylabel("Preço de fechamento")
    ax.set_title("Preço de fechamento BVSP por ano", weight="bold")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter(PRICE_FORMATTER)

    fig.tight_layout()
    plt.show()


def plot_average_close_price_per_year(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot = df_plot.loc["2001-01-02":"2023-12-28"]
    df_plot["year"] = df_plot.index.year
    df_plot = df_plot.groupby("year").mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim([0, 150000])

    sns.lineplot(x="year", y=Col.close, data=df_plot)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Preço médio de fechamento")
    ax.set_title("Preço médio de fechamento BVSP por ano", weight="bold")

    plt.xticks(list(range(2001, 2024)), rotation=45)
    ax.yaxis.set_major_formatter(PRICE_FORMATTER)

    fig.tight_layout()
    plt.show()


def plot_variables_correlation(df: pd.DataFrame) -> None:
    df_plot = df.drop(Col.var, axis=1)
    corr = df_plot.corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(data=corr, annot=True, linewidths=0.5, ax=ax)

    ax.set_title("Correlação entre variáveis", weight="bold")
    fig.tight_layout()
    plt.show()


def plot_box_violin(df: pd.DataFrame) -> None:
    df_plot = df.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(x=Col.var, data=df_plot, ax=ax, whis=1.5, color="lightblue")
    sns.violinplot(x=Col.var, data=df_plot, ax=ax, color="lightgray")

    ax.set_title("Distribuição da variação do preço", weight="bold")
    ax.set_xlabel("Variação percentual do preço diária")

    ax.xaxis.set_major_formatter(VARIATION_FOMATTER)

    fig.tight_layout()
    plt.show()


def plot_kde(df: pd.DataFrame) -> None:
    df_plot = df.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(data=df_plot[Col.var], ax=ax, fill=True)

    ax.set_title("Distribuição da variação do preço", weight="bold")
    ax.set_xlabel("Variação percentual do preço diária")
    ax.set_ylabel("Frequência")

    plt.xticks(list(range(-15, 16, 3)))

    ax.xaxis.set_major_formatter(VARIATION_FOMATTER)

    fig.tight_layout()
    plt.show()


def plot_variation(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot = df_plot.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x=Col.date, y=Col.var, data=df_plot)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Variação percentual do preço")
    ax.set_title("Variação percentual do preço BVSP", weight="bold")

    ax.yaxis.set_major_formatter(VARIATION_FOMATTER)

    ax.xaxis.set_major_locator(YearLocator())
    plt.xticks(rotation=45)

    fig.tight_layout()
    plt.show()


def plot_seasonal_decompose(
    df: pd.DataFrame,
    column: str,
    period: int,
    formatter: ticker.FuncFormatter = None,
) -> None:
    df_plot = df.copy()

    column_to_drop = list(
        {Col.close, Col.max, Col.min, Col.open, Col.vol, Col.var} - {column},
    )
    df_plot = df_plot.drop(column_to_drop, axis=1)
    df_plot = df_plot.dropna()

    decomposed = seasonal_decompose(df_plot, period=period)

    _, ax = plt.subplots(3, 1, figsize=(16, 7))

    ax[0].set_title("Tendência")
    decomposed.trend.plot(ax=ax[0])
    ax[1].set_title("Sazonalidade")
    decomposed.seasonal.plot(ax=ax[1])
    ax[2].set_title("Resíduos")
    decomposed.resid.plot(ax=ax[2])

    for axis in ax:
        axis.xaxis.set_major_locator(YearLocator())
        axis.yaxis.set_major_formatter(formatter)
        axis.set_xlabel("")

    plt.tight_layout()
    plt.show()


def plot_outliers_detection(df: pd.DataFrame) -> None:
    df_plot = df.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(x=Col.date, y=Col.var, data=df_plot, ax=ax, color="lightblue")
    ax.set_ylabel("Variação percentual do preço")
    ax.set_xlabel("Data")
    ax.set_title("Preço de fechamento com variação percentual BVSP", weight="bold")
    ax.yaxis.set_major_formatter(VARIATION_FOMATTER)

    ax2 = ax.twinx()

    ax2.set_ylim([0, 150000])
    sns.lineplot(x=Col.date, y=Col.close, data=df_plot, ax=ax2)
    ax2.set_ylabel("Preço de fechamento")
    ax2.yaxis.set_major_formatter(PRICE_FORMATTER)

    fig.tight_layout()
    plt.grid(visible=False)
    plt.show()


if __name__ == "__main__":
    main()
