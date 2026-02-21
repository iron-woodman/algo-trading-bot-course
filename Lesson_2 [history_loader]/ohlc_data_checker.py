"""Модуль для проверки целостности данных OHLCV с использованием Polars."""

from typing import Optional, Any, Tuple, TypeAlias
import polars as pl
from datetime import datetime, timezone, timedelta

# Определение типа для отчета
ReportDict: TypeAlias = dict[str, Any]


def _get_tf_timedelta(timeframe: str) -> timedelta:
    """Преобразование таймфрейма в объект timedelta."""
    tf: str = timeframe.lower().strip()
    if tf.endswith('m'):
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith('h'):
        return timedelta(hours=int(tf[:-1]))
    if tf.endswith('d'):
        return timedelta(days=int(tf[:-1]))
    raise ValueError(f"Неподдерживаемый формат таймфрейма: {timeframe}")


def check_ohlcv_gaps(
    df: pl.DataFrame,
    timeframe: str = "5m",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    ts_col: str = "timestamp",
    datetime_col: str = "datetime",
    ohlcv_cols: Optional[list[str]] = None,
) -> Tuple[ReportDict, pl.DataFrame]:
    """
    Выполнение проверки DataFrame с OHLCV на пропуски, дубликаты и корректность порядка времени.

    Args:
        df (pl.DataFrame): DataFrame с данными OHLCV.
        timeframe (str, optional): Таймфрейм свечей. По умолчанию "5m".
        start (Optional[datetime], optional): Начальная граница проверки. По умолчанию None.
        end (Optional[datetime], optional): Конечная граница проверки. По умолчанию None.
        ts_col (str, optional): Имя колонки с меткой времени. По умолчанию "timestamp".
        datetime_col (str, optional): Имя колонки с datetime. По умолчанию "datetime".
        ohlcv_cols (Optional[list[str]], optional): Список колонок для проверки на NaN. По умолчанию None.

    Returns:
        Tuple[ReportDict, pl.DataFrame]: Кортеж, содержащий словарь с отчетом и DataFrame с пропусками.
    """
    if ohlcv_cols is None:
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    if df.is_empty():
        return {
            'timeframe': timeframe,
            'expected_points': 0,
            'present_points': 0,
            'missing_points': 0,
            'percent_missing': 0.0,
            'num_gaps': 0,
            'longest_gap': None,
            'duplicated_index_count': 0,
            'unsorted_count': 0,
            'partial_column_nans': {},
            'start_expected': None,
            'end_expected': None,
        }, pl.DataFrame()

    # Подготовка данных
    df_work = df.clone()
    
    # Преобразование timestamp в datetime, если нужно
    if datetime_col not in df_work.columns and ts_col in df_work.columns:
        df_work = df_work.with_columns(
            pl.from_epoch(pl.col(ts_col), time_unit="ms").dt.replace_time_zone("UTC").alias(datetime_col)
        )
    elif datetime_col in df_work.columns:
        df_work = df_work.with_columns(pl.col(datetime_col).dt.replace_time_zone("UTC"))

    # Проверка на сортировку
    unsorted_count = df_work.filter(pl.col(datetime_col).diff() < timedelta(0)).height
    if unsorted_count > 0:
        df_work = df_work.sort(datetime_col)

    # Подсчет дубликатов
    duplicated_count = df_work.select(pl.col(datetime_col)).is_duplicated().sum()
    df_unique = df_work.unique(subset=[datetime_col]).sort(datetime_col)

    # Определение границ
    actual_start = df_unique[datetime_col].min()
    actual_end = df_unique[datetime_col].max()
    
    check_start = start if start else actual_start
    check_end = end if end else actual_end

    if check_start > check_end:
        raise ValueError(f"Начало ({check_start}) позже конца ({check_end})")

    # Создание эталонной сетки
    expected_grid = pl.datetime_range(
        start=check_start,
        end=check_end,
        interval=timeframe.lower().strip(),
        time_zone="UTC",
        eager=True
    ).to_frame(datetime_col)

    # Join с эталонной сеткой
    # Приводим к одной точности (микросекунды), так как Polars строго проверяет это при join
    df_unique = df_unique.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
    expected_grid = expected_grid.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
    
    reindexed = expected_grid.join(df_unique, on=datetime_col, how="left")
    
    # Поиск пропусков
    # Считаем строку пропущенной, если все OHLCV колонки равны null
    missing_mask = reindexed.select(
        pl.all_horizontal(pl.col(ohlcv_cols).is_null()).alias("is_missing")
    )
    reindexed = reindexed.with_columns(missing_mask)
    
    missing_count = reindexed["is_missing"].sum()
    total_expected = len(expected_grid)
    percent_missing = (missing_count / total_expected * 100) if total_expected > 0 else 0.0

    # Группировка пропусков
    gaps_df = pl.DataFrame()
    tf_delta = _get_tf_timedelta(timeframe)
    if missing_count > 0:
        # Алгоритм группировки последовательных значений
        gaps_df = (
            reindexed
            .with_columns([
                (pl.col("is_missing") != pl.col("is_missing").shift(1)).fill_null(True).cum_sum().alias("group")
            ])
            .filter(pl.col("is_missing"))
            .group_by("group")
            .agg([
                pl.col(datetime_col).min().alias("gap_start"),
                pl.col(datetime_col).max().alias("gap_end"),
                pl.count().alias("missing_count")
            ])
            .with_columns([
                (pl.col("gap_end") - pl.col("gap_start") + tf_delta).alias("duration")
            ])
            .sort("gap_start")
        )

    # Проверка на NaN (null в polars) в отдельных колонках
    col_nulls = {col: reindexed[col].is_null().sum() for col in ohlcv_cols}

    longest_gap = None
    if not gaps_df.is_empty():
        max_row = gaps_df.sort("missing_count", descending=True).head(1).to_dicts()[0]
        longest_gap = {
            'gap_start': max_row['gap_start'],
            'gap_end': max_row['gap_end'],
            'missing_count': max_row['missing_count'],
            'duration': max_row['duration']
        }

    report: ReportDict = {
        'timeframe': timeframe,
        'expected_points': total_expected,
        'present_points': total_expected - missing_count,
        'missing_points': missing_count,
        'percent_missing': percent_missing,
        'num_gaps': len(gaps_df),
        'longest_gap': longest_gap,
        'duplicated_index_count': duplicated_count,
        'unsorted_count': unsorted_count,
        'partial_column_nans': col_nulls,
        'start_expected': expected_grid[datetime_col].min(),
        'end_expected': expected_grid[datetime_col].max(),
    }

    return report, gaps_df
