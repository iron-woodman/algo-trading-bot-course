"""Модуль для проверки целостности данных OHLCV."""

from typing import Optional, Any, Tuple, TypeAlias
import pandas as pd
import numpy as np

# Определение типа для отчета
ReportDict: TypeAlias = dict[str, Any]


def _tf_to_pandas_freq(timeframe: str) -> str:
    """
    Преобразование формата timeframe в частоту pandas.

    Args:
        timeframe (str): Таймфрейм (например, '5m', '1h', '1d').

    Returns:
        str: Частота для pandas (например, '5T', '1H', '1D').

    Raises:
        ValueError: Возникает при использовании неподдерживаемого формата таймфрейма.

    Example:
        >>> _tf_to_pandas_freq('5m')
        '5T'
    """
    tf: str = timeframe.lower().strip()
    if tf.endswith('m'):
        return f"{int(tf[:-1])}min"
    if tf.endswith('h'):
        return f"{int(tf[:-1])}h"
    if tf.endswith('d'):
        return f"{int(tf[:-1])}d"
    raise ValueError(f"Неподдерживаемый формат таймфрейма: {timeframe}")


def check_ohlcv_gaps(
    df: pd.DataFrame,
    timeframe: str = "5m",
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    ts_col: str = "timestamp",
    datetime_col: Optional[str] = None,
    ohlcv_cols: Optional[list[str]] = None,
) -> Tuple[ReportDict, pd.DataFrame]:
    """
    Выполнение проверки DataFrame с OHLCV на пропуски, дубликаты, NaN и корректность порядка времени.

    Args:
        df (pd.DataFrame): DataFrame с данными OHLCV.
        timeframe (str, optional): Таймфрейм свечей. По умолчанию "5m".
        start (Optional[pd.Timestamp | str], optional): Начальная граница проверки. По умолчанию None.
        end (Optional[pd.Timestamp | str], optional): Конечная граница проверки. По умолчанию None.
        ts_col (str, optional): Имя колонки с меткой времени в мс. По умолчанию "timestamp".
        datetime_col (Optional[str], optional): Имя колонки с datetime. По умолчанию None.
        ohlcv_cols (Optional[list[str]], optional): Список колонок для проверки на NaN. По умолчанию None.

    Returns:
        Tuple[ReportDict, pd.DataFrame]: Кортеж, содержащий словарь с отчетом и DataFrame с пропусками.

    Raises:
        ValueError: Возникает при отсутствии временных колонок или некорректных границах интервала.

    Example:
        >>> report, gaps = check_ohlcv_gaps(df, timeframe='1h')
    """
    if ohlcv_cols is None:
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    df_work: pd.DataFrame = df.copy()

    # Установка временного индекса (UTC)
    if datetime_col and datetime_col in df_work.columns:
        df_work[datetime_col] = pd.to_datetime(df_work[datetime_col], utc=True)
        df_work = df_work.set_index(datetime_col)
    elif ts_col in df_work.columns:
        # Ожидается время в миллисекундах
        df_work[ts_col] = pd.to_datetime(df_work[ts_col], unit='ms', utc=True)
        df_work = df_work.set_index(ts_col)
    else:
        # Проверка существующего индекса
        if not isinstance(df_work.index, pd.DatetimeIndex):
            raise ValueError("Отсутствует колонка времени и индекс не является DateTimeIndex.")
        # Локализация индекса в UTC
        if df_work.index.tzinfo is None:
            df_work.index = df_work.index.tz_localize('UTC')
        else:
            df_work.index = df_work.index.tz_convert('UTC')

    # Проверка и сортировка индекса
    unsorted_mask: pd.Series = (df_work.index.to_series().diff().fillna(pd.Timedelta(seconds=1)) <= pd.Timedelta(0))
    unsorted_count: int = int(unsorted_mask.sum())
    if unsorted_count > 0:
        df_work = df_work.sort_index()

    # Подсчет дубликатов по индексу
    duplicated_index_count: int = int(df_work.index.duplicated().sum())
    # Удаление дубликатов для корректного построения ожидаемой последовательности
    df_unique: pd.DataFrame = df_work[~df_work.index.duplicated(keep='first')]

    # Определение границ проверки
    if start is None:
        if df_unique.empty:
            return {
                'timeframe': timeframe,
                'expected_points': 0,
                'present_points': 0,
                'missing_points': 0,
                'percent_missing': 0.0,
                'num_gaps': 0,
                'longest_gap': None,
                'duplicated_index_count': duplicated_index_count,
                'unsorted_count': unsorted_count,
                'partial_column_nans': {},
                'start_expected': None,
                'end_expected': None,
            }, pd.DataFrame()
        start_ts: pd.Timestamp = df_unique.index.min()
    else:
        start_ts = pd.to_datetime(start).tz_localize('UTC') if pd.to_datetime(start).tzinfo is None else pd.to_datetime(start).tz_convert('UTC')
    
    if end is None:
        if df_unique.empty:
             return {
                'timeframe': timeframe,
                'expected_points': 0,
                'present_points': 0,
                'missing_points': 0,
                'percent_missing': 0.0,
                'num_gaps': 0,
                'longest_gap': None,
                'duplicated_index_count': duplicated_index_count,
                'unsorted_count': unsorted_count,
                'partial_column_nans': {},
                'start_expected': None,
                'end_expected': None,
            }, pd.DataFrame()
        end_ts: pd.Timestamp = df_unique.index.max()
    else:
        end_ts = pd.to_datetime(end).tz_localize('UTC') if pd.to_datetime(end).tzinfo is None else pd.to_datetime(end).tz_convert('UTC')

    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Не удалось определить границы интервала (start/end являются NaT).")

    if start_ts >= end_ts:
        raise ValueError(f"Начало интервала ({start_ts}) должно быть меньше его конца ({end_ts}).")

    # Создание эталонной последовательности временных меток
    freq: str = _tf_to_pandas_freq(timeframe)
    expected_index: pd.DatetimeIndex = pd.date_range(start=start_ts, end=end_ts, freq=freq, tz='UTC')

    # Реиндексация для выявления пропущенных свечей
    reindexed: pd.DataFrame = df_unique.reindex(expected_index)

    # Идентификация отсутствующих данных (все колонки OHLCV содержат NaN)
    missing_mask: pd.Series = reindexed[ohlcv_cols].isna().all(axis=1)
    missing_count: int = int(missing_mask.sum())
    total_expected: int = len(expected_index)
    percent_missing: float = (missing_count / total_expected) * 100.0 if total_expected > 0 else 0.0

    # Группировка последовательных пропусков в блоки
    gaps: list[dict[str, Any]] = []
    if missing_count > 0:
        is_missing: np.ndarray = missing_mask.astype(int).values
        idx: pd.DatetimeIndex = reindexed.index
        i: int = 0
        n: int = len(is_missing)
        while i < n:
            if is_missing[i]:
                j: int = i
                while j + 1 < n and is_missing[j + 1]:
                    j += 1
                gap_start: pd.Timestamp = idx[i]
                gap_end: pd.Timestamp = idx[j]
                gap_count: int = j - i + 1
                gap_duration: pd.Timedelta = (gap_end - gap_start) + pd.to_timedelta(freq)
                gaps.append({
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'missing_count': gap_count,
                    'duration': gap_duration
                })
                i = j + 1
            else:
                i += 1

    # Проверка наличия NaN в отдельных колонках
    col_nans: dict[str, int] = {col: int(reindexed[col].isna().sum()) for col in ohlcv_cols if col in reindexed.columns}

    # Поиск самого длинного периода отсутствия данных
    longest_gap: Optional[dict[str, Any]] = max(gaps, key=lambda x: x['missing_count']) if gaps else None

    # Формирование итогового отчета
    report: ReportDict = {
        'timeframe': timeframe,
        'expected_points': total_expected,
        'present_points': int(total_expected - missing_count),
        'missing_points': missing_count,
        'percent_missing': percent_missing,
        'num_gaps': len(gaps),
        'longest_gap': longest_gap,
        'duplicated_index_count': duplicated_index_count,
        'unsorted_count': unsorted_count,
        'partial_column_nans': col_nans,
        'start_expected': expected_index[0] if total_expected > 0 else None,
        'end_expected': expected_index[-1] if total_expected > 0 else None,
    }

    return report, pd.DataFrame(gaps)
