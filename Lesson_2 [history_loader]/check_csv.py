"""Утилита для проверки целостности данных OHLCV в CSV файле."""

import pandas as pd
import sys
from typing import Optional
from ohlc_data_checker import check_ohlcv_gaps


def check_file(file_path: str, timeframe: str = '5m') -> None:
    """
    Выполнение проверки CSV файла на наличие пропусков и дубликатов.

    Args:
        file_path (str): Путь к CSV файлу.
        timeframe (str, optional): Таймфрейм свечей. По умолчанию '5m'.

    Returns:
        None
    """
    try:
        # Загрузка данных
        print(f"Чтение файла: {file_path}")
        df: pd.DataFrame = pd.read_csv(file_path)

        if 'timestamp' not in df.columns and 'datetime' not in df.columns:
            print("Ошибка: В файле должны быть колонки 'timestamp' или 'datetime'.")
            return

        # Определение параметров времени
        datetime_col: Optional[str] = 'datetime' if 'datetime' in df.columns else None
        ts_col: str = 'timestamp'

        # Запуск проверки
        report, gaps_df = check_ohlcv_gaps(
            df,
            timeframe=timeframe,
            ts_col=ts_col,
            datetime_col=datetime_col
        )

        print("\n--- Результаты проверки ---")
        for key, value in report.items():
            if key == 'longest_gap' and value:
                print(f"{key}: {value['missing_count']} свечей (с {value['gap_start']} по {value['gap_end']})")
            else:
                print(f"{key}: {value}")

        if not gaps_df.empty:
            print("\nОбнаружены блоки пропусков:")
            print(gaps_df[['gap_start', 'gap_end', 'missing_count', 'duration']])
        else:
            print("\nПропусков в данных не обнаружено.")

    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python check_csv.py <путь_к_файлу> [таймфрейм]")
        print("Пример: python check_csv.py data/ETH_USDT_5m_2025.csv 5m")
    else:
        path: str = sys.argv[1]
        tf: str = sys.argv[2] if len(sys.argv) > 2 else '5m'
        check_file(path, tf)
