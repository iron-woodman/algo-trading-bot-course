"""Утилита для проверки целостности данных OHLCV в Parquet файле с использованием Polars."""

import polars as pl
import sys
from typing import Optional
from ohlc_data_checker import check_ohlcv_gaps


def check_file(file_path: str, timeframe: str = '5m') -> None:
    """
    Выполнение проверки Parquet файла на наличие пропусков и дубликатов.

    Args:
        file_path (str): Путь к Parquet файлу.
        timeframe (str, optional): Таймфрейм свечей. По умолчанию '5m'.

    Returns:
        None
    """
    try:
        # Загрузка данных
        print(f"Чтение файла: {file_path}")
        df: pl.DataFrame = pl.read_parquet(file_path)

        if 'timestamp' not in df.columns and 'datetime' not in df.columns:
            print("Ошибка: В файле должны быть колонки 'timestamp' или 'datetime'.")
            return

        # Запуск проверки
        report, gaps_df = check_ohlcv_gaps(
            df,
            timeframe=timeframe,
            ts_col='timestamp',
            datetime_col='datetime'
        )

        print("
--- Результаты проверки ---")
        for key, value in report.items():
            if key == 'longest_gap' and value:
                print(f"{key}: {value['missing_count']} свечей (с {value['gap_start']} по {value['gap_end']})")
            elif key == 'partial_column_nans':
                 print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")

        if not gaps_df.is_empty():
            print("
Обнаружены блоки пропусков:")
            # Выводим первые 20 пропусков для компактности
            print(gaps_df.head(20))
        else:
            print("
Пропусков в данных не обнаружено.")

    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python check_parquet.py <путь_к_файлу> [таймфрейм]")
        print("Пример: python check_parquet.py data/ETH_USDT_30m_2025.parquet 30m")
    else:
        path: str = sys.argv[1]
        tf: str = sys.argv[2] if len(sys.argv) > 2 else '30m'
        check_file(path, tf)
